import torch 
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class LLaMAGridTransformer(nn.Module):
    def __init__(self, model_name="facebook/opt-350m", out_channels=10):
        super().__init__()
        self.tokenizer    = AutoTokenizer.from_pretrained(model_name)
        self.model        = AutoModel.from_pretrained(
                               model_name,
                               torch_dtype=torch.float16
                                          if torch.cuda.is_available()
                                          else torch.float32
                           )
        self.out_channels = out_channels
        self.proj         = None        # ← don’t build until we know real_dim

    def forward(self, grid):  # grid: (B, C, H, W)
        B, C, H, W = grid.shape

        # flatten into token IDs
        flat_ids = grid.argmax(dim=1).view(B, -1)   # (B, H*W)
        text_in  = [" ".join(map(str, row.tolist())) for row in flat_ids]
        tokens   = self.tokenizer(
                       text_in,
                       return_tensors="pt",
                       padding=True,
                       truncation=True
                   ).to(grid.device)

        # run the transformer
        hidden   = self.model(**tokens).last_hidden_state  # (B, seq_len, real_dim)
        real_dim = hidden.shape[-1]

        # lazy‐init proj ONCE
        if self.proj is None:
            self.proj = nn.Linear(real_dim, self.out_channels) \
                            .to(hidden.dtype).to(hidden.device)

        # project and reshape
        logits = self.proj(hidden)              # (B, seq_len, out_channels)
        logits = logits[:, : H*W, :]            # crop to H*W tokens
        logits = logits.transpose(1, 2)         # → (B, out_channels, seq_len)
        logits = logits.reshape(B, C, H, W)     # → (B, C, H, W)

        return logits

import torch
import torch.nn as nn

class SimpleGridTransformer(nn.Module):
    def __init__(self, num_colors=10, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.d_model = d_model
        # project one‑hot pixels into d_model embeddings
        self.input_proj = nn.Linear(num_colors, d_model)
        # learned 2D positional encoding for H×W tokens
        self.pos_enc = nn.Parameter(torch.randn(1, d_model, 1, 1))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4*d_model),
            num_layers
        )
        # project back to color logits
        self.output_proj = nn.Linear(d_model, num_colors)

    def forward(self, x):
        """
        x: (B, C=num_colors, H, W) one‑hot or soft logits
        returns: (B, num_colors, H, W)
        """
        B, C, H, W = x.shape
        # flatten spatial → (B, H*W, C)
        tokens = x.view(B, C, -1).permute(0, 2, 1)
        # embed colors → (B, H*W, d_model)
        emb = self.input_proj(tokens)
        # add a simple broadcast pos_enc (you could also make 2D)
        emb = emb + self.pos_enc.view(1, self.d_model, 1).permute(0,2,1)
        # transformer expects (B, S, E)
        out = self.transformer(emb)
        # back to color logits
        logits = self.output_proj(out)           # (B, H*W, num_colors)
        # reshape → (B, num_colors, H, W)
        return logits.permute(0,2,1).view(B, C, H, W)
    
class GridFusionModel(nn.Module):
    def __init__(self, cnn_model, llama_model, out_channels=10):
        super().__init__()
        self.cnn = cnn_model
        self.llama = llama_model
        self.fuse = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)

    def forward(self, x):  # x: (B, C=10, H, W)
        cnn_out = self.cnn(x)        # (B, 10, H, W)
        llama_out = self.llama(x)    # (B, 10, H, W)

        fused = torch.cat([cnn_out, llama_out], dim=1)  # (B, 20, H, W)
        out = self.fuse(fused)  # (B, 10, H, W)
        return out

class PatchSeqTransformer(nn.Module):
    def __init__(self,
                 num_colors:  int,   # e.g. 10
                 branch_ch:   int,   # e.g. 16
                 max_patches: int,   # e.g. 900
                 nhead:       int = 4,
                 num_layers:  int = 2):
        super().__init__()
        P = 11
        embed_dim = branch_ch * 3

        # ── 1) multi‐scale CNNs ──────────────────────────
        self.conv3 = nn.Sequential(
            nn.Conv2d(num_colors, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, branch_ch, kernel_size=1),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(num_colors, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, branch_ch, kernel_size=1),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(num_colors, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv2d(32, branch_ch, kernel_size=1),
        )

        # ── 2) positional encoding ───────────────────────
        self.pos_enc = nn.Parameter(torch.zeros(1, max_patches, embed_dim))

        # ── 3) transformer ───────────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # ── 4) project each token back into a full patch ─
        self.to_patch = nn.Linear(embed_dim, num_colors * P * P)

        self.P    = P
        self.half = P // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, S, C, P, P) → returns (B, C, P, P) when S=1,
                            or (B, S, C, P, P) for longer sequences.
        """
        # If you’re still feeding it 4‑D [B,C,P,P], just do:
        if x.dim() == 4:
            x = x.unsqueeze(1)               # → (B,1,C,P,P)

        B, S, C, P, _ = x.shape             # now always 5‑D

        # 1) flatten and CNN
        x = x.view(B * S, C, P, P)          # (B·S, C, P, P)
        f3 = self.conv3(x)
        f5 = self.conv5(x)
        f7 = self.conv7(x)
        feats = torch.cat([f3, f5, f7], dim=1)  # (B·S, embed_dim, P, P)

        # 2) center‑pixel embeddings → tokens
        tokens = feats[:, :, self.half, self.half]  # (B·S, embed_dim)
        tokens = tokens.view(B, S, -1)              # (B, S, embed_dim)

        # 3) add pos‑enc & run transformer
        tokens = tokens + self.pos_enc[:, :S, :]
        out    = self.transformer(tokens)           # (B, S, embed_dim)

        # 4) project *each* token into its full patch
        patch_flat = self.to_patch(out)             # (B, S, C*P*P)
        patches    = patch_flat.view(B, S, 
                                     C, P, P)       # (B, S, C, P, P)

        # if you only ever pass S=1, just drop that dim:
        if patches.size(1) == 1:
            return patches.squeeze(1)               # → (B, C, P, P)
        return patches                              # → (B, S, C, P, P)


class MultiScaleTransformerTranslator(nn.Module):
    def __init__(self,
                 num_colors:   int,   # e.g. 10 for ARC
                 branch_ch:    int,   # channels per CNN branch (e.g. 16)
                 max_patches:  int,   # maximum H*W you expect
                 nhead:        int=4,
                 num_layers:   int=2):
        super().__init__()
        embed_dim = branch_ch * 3

        # ── 1) three CNN “patch‐views” ───────────────────────────────────
        self.conv3 = nn.Sequential(
            nn.Conv2d(num_colors, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, branch_ch, kernel_size=1)  # → branch_ch channels
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(num_colors, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, branch_ch, kernel_size=1)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(num_colors, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv2d(32, branch_ch, kernel_size=1)
        )

        # ── 2) learned positional encoding for seq‐length ≤ max_patches ──
        #   (1, max_patches, embed_dim)
        self.pos_enc = nn.Parameter(torch.zeros(1, max_patches, embed_dim))

        # ── 3) small Transformer encoder ────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=num_layers)

        # ── 4) project each token back to num_colors logits ─────────────
        self.to_logits = nn.Linear(embed_dim, num_colors)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: one‑hot input of shape (B, C=num_colors, H, W)
        returns: per‑pixel logits (B, C, H, W)
        """
        B, C, H, W = x.shape

        # 1) multi‑scale CNNs → three (B, branch_ch, H, W)
        f3 = self.conv3(x)
        f5 = self.conv5(x)
        f7 = self.conv7(x)

        # 2) fuse them into one (B, embed_dim, H, W)
        feats = torch.cat([f3, f5, f7], dim=1)

        # 3) flatten spatial → sequence: (B, patches=H*W, embed_dim)
        seq = feats.view(B, -1, H*W).permute(0, 2, 1)

        # 4) add positional enc (truncate if H*W < max_patches)
        seq = seq + self.pos_enc[:, : H*W, :]

        # 5) Transformer refinement
        seq = self.transformer(seq)  # (B, H*W, embed_dim)

        # 6) project each token → color‑logits: (B, H*W, num_colors)
        seq = self.to_logits(seq)

        # 7) reshape back to (B, num_colors, H, W)
        out = seq.permute(0, 2, 1).view(B, C, H, W)
        return out

class MultiScalePatchTranslator(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        
        # Each CNN handles a different patch size
        self.conv3 = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=1)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=1)
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=1)
        )

        # Final fusion layer
        self.final = nn.Conv2d(16 * 3, num_channels, kernel_size=1)

    def forward(self, x):
        out3 = self.conv3(x)
        out5 = self.conv5(x)
        out7 = self.conv7(x)

        # Concatenate along channel dimension
        combined = torch.cat([out3, out5, out7], dim=1)
        return self.final(combined)

class PatchTransformer(nn.Module):
    def __init__(self, num_colors, num_patches, embed_dim, num_heads, num_layers):
        super().__init__()
        self.positional_encoding = nn.Parameter(torch.randn(1, num_patches, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                   nhead=num_heads,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.to_output = nn.Linear(embed_dim, num_colors)

    def forward(self, x):  # x shape: (B, num_patches, embed_dim)
        x = x + self.positional_encoding
        x = self.transformer(x)
        return self.to_output(x.mean(dim=1))  # global average or CLS token

# ── 4) Define CNN translator ───────────────────────────────────────────────────
class PatchTranslatorModel(nn.Module):
    def __init__(self, num_channels=10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, num_channels, kernel_size=1),
        )
    def forward(self, x):
        return self.network(x)