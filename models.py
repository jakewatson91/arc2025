import torch 
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class PatchCNN(nn.Module):
    def __init__(self, num_colors=10): 
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_colors, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),        nn.ReLU(),
            nn.Conv2d(32, num_colors, 1),
        )
    def forward(self, x): return self.net(x)

class GridTransformer(nn.Module):
    def __init__(self, num_colors, H, W, embed_dim=64, nhead=4, num_layers=2):
        super().__init__()
        self.proj_in  = nn.Conv2d(num_colors, embed_dim, 1)
        self.pos_enc  = nn.Parameter(torch.zeros(1, H*W, embed_dim))
        enc_layer     = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.proj_out = nn.Linear(embed_dim, num_colors)
    
    def forward(self, x):
        B, C, H, W = x.shape
        f = self.proj_in(x)                       # (B, E, H, W)
        seq = f.view(B, f.shape[1], -1).permute(0,2,1)  # (B, H*W, E)
        seq = seq + self.pos_enc[:, :H*W, :]
        seq = self.transformer(seq)               # (B, H*W, E)
        logits = self.proj_out(seq)               # (B, H*W, C)
        out = logits.permute(0,2,1).view(B, C, H, W)
        return out

class FusionModel(nn.Module):
    def __init__(self, patch_model, grid_model, num_colors=10):
        super().__init__()
        self.patch = patch_model
        self.grid  = grid_model
        self.fuse  = nn.Conv2d(num_colors*2, num_colors, 1)
    
    def forward(self, x):
        p = self.patch(x)  # (B, C, H, W)
        g = self.grid(x)   # (B, C, H, W)
        return self.fuse(torch.cat([p,g], dim=1))
