import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# ── Hyperparameters ────────────────────────────────────────────────────────────
PATCH_SIZE    = 11
NUM_COLORS    = 10
LR            = 1e-5
EPOCHS        = 1000
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HALF_PATCH    = PATCH_SIZE // 2
PAD_VAL       = -1

# ── One‑hot encoding helper ────────────────────────────────────────────────────
def to_onehot(grid: np.ndarray, num_classes: int = NUM_COLORS) -> torch.Tensor:
    eye    = np.eye(num_classes, dtype=np.float32)
    onehot = eye[grid]
    return torch.from_numpy(onehot.transpose(2,0,1))

# ── 1) Load a single ARC task ──────────────────────────────────────────────────
with open("data/arc-agi_training_challenges.json") as f:
    train_tasks = json.load(f)
with open("data/arc-agi_training_solutions.json") as f:
    train_sols  = json.load(f)

# DEBUG: pick just the first task
task_id = next(iter(train_tasks))
print(f"[DEBUG] Using only task {task_id}")
inp0 = np.array(train_tasks[task_id]['train'][0]['input'])
out0 = np.array(train_sols[task_id][0])

# compute canvas dims
H, W   = inp0.shape
Oh, Ow = out0.shape
Ch     = max(H, Oh)
Cw     = max(W, Ow)

# pad input
pad_top_in    = HALF_PATCH
pad_bottom_in = HALF_PATCH + (Ch - H)
pad_left_in   = HALF_PATCH
pad_right_in  = HALF_PATCH + (Cw - W)
padded_in  = np.pad(
    inp0,
    ((pad_top_in, pad_bottom_in), (pad_left_in, pad_right_in)),
    constant_values=PAD_VAL
)

# pad output (use Ch, Ow)
pad_top_out    = HALF_PATCH
pad_bottom_out = HALF_PATCH + (Ch - Oh)
pad_left_out   = HALF_PATCH
pad_right_out  = HALF_PATCH + (Cw - Ow)
padded_out = np.pad(
    out0,
    ((pad_top_out, pad_bottom_out), (pad_left_out, pad_right_out)),
    constant_values=PAD_VAL
)

assert padded_in.shape == padded_out.shape, "Padding mismatch!"
print(f"[DEBUG] padded shape: {padded_in.shape}")

# tensors
x_full = to_onehot(padded_in).unsqueeze(0).to(DEVICE)        # (1,C,Hp,Wp)
y_full = torch.from_numpy(padded_out).long().unsqueeze(0).to(DEVICE)  # (1,Hp,Wp)
_, C, Hp, Wp = x_full.shape

# ── 2) Models ─────────────────────────────────────────────────────────────────
class PatchCNN(nn.Module):
    def __init__(self, num_colors): 
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_colors, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),        nn.ReLU(),
            nn.Conv2d(32, num_colors, 1),
        )
    def forward(self, x): return self.net(x)

class GridTransformer(nn.Module):
    def __init__(self, num_colors, embed_dim=64, nhead=4, num_layers=2):
        super().__init__()
        self.proj_in  = nn.Conv2d(num_colors, embed_dim, 1)
        self.pos_enc  = nn.Parameter(torch.zeros(1, Hp*Wp, embed_dim))
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
    def __init__(self, patch_model, grid_model):
        super().__init__()
        self.patch = patch_model
        self.grid  = grid_model
        self.fuse  = nn.Conv2d(NUM_COLORS*2, NUM_COLORS, 1)
    def forward(self, x):
        p = self.patch(x)  # (B, C, H, W)
        g = self.grid(x)   # (B, C, H, W)
        return self.fuse(torch.cat([p,g], dim=1))

patch_cnn = PatchCNN(NUM_COLORS).to(DEVICE)
grid_tr   = GridTransformer(NUM_COLORS).to(DEVICE)
fusion    = FusionModel(patch_cnn, grid_tr).to(DEVICE)

# ── 3) Loss & optimizers ───────────────────────────────────────────────────────
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_VAL)
opt_p   = torch.optim.Adam(patch_cnn.parameters(), lr=LR)
opt_g   = torch.optim.Adam(grid_tr.parameters(),   lr=LR)
opt_f   = torch.optim.Adam(fusion.parameters(),    lr=LR)

# ── 4) Train PatchCNN ─────────────────────────────────────────────────────────
print("\n🚀 Training PatchCNN")
for epoch in range(1, EPOCHS+1):
    patch_cnn.train()
    opt_p.zero_grad()
    logits = patch_cnn(x_full)                 # (1,C,Hp,Wp)
    flat_logits = logits.permute(0,2,3,1).reshape(-1,C)
    flat_labels = y_full.reshape(-1)
    loss = loss_fn(flat_logits, flat_labels)
    print(f"[PatchCNN] epoch {epoch}, loss={loss.item():.4f}")
    loss.backward()
    opt_p.step()

# ── 5) Train GridTransformer ──────────────────────────────────────────────────
print("\n🚀 Training GridTransformer")
for epoch in range(1, EPOCHS+1):
    grid_tr.train()
    opt_g.zero_grad()
    logits = grid_tr(x_full)
    flat_logits = logits.permute(0,2,3,1).reshape(-1,C)
    flat_labels = y_full.reshape(-1)
    loss = loss_fn(flat_logits, flat_labels)
    print(f"[GridTrans] epoch {epoch}, loss={loss.item():.4f}")
    loss.backward()
    opt_g.step()

# ── 6) Train FusionModel ──────────────────────────────────────────────────────
print("\n🚀 Training FusionModel")
for epoch in range(1, EPOCHS+1):
    fusion.train()
    opt_f.zero_grad()
    logits = fusion(x_full)
    flat_logits = logits.permute(0,2,3,1).reshape(-1,C)
    flat_labels = y_full.reshape(-1)
    loss = loss_fn(flat_logits, flat_labels)
    print(f"[Fusion] epoch {epoch}, loss={loss.item():.4f}")
    loss.backward()
    opt_f.step()

# ── 7) Final debug & accuracy ─────────────────────────────────────────────────
fusion.eval()
with torch.no_grad():
    pred = fusion(x_full).argmax(1)[0].cpu().numpy()
    cropped_pred = pred[HALF_PATCH: Oh + HALF_PATCH, HALF_PATCH: Ow + HALF_PATCH]

print(f"\n[DEBUG] Predicted grid:\n{cropped_pred}")
print(f"[DEBUG] Ground truth:\n{out0}")

# exact match?
grid_match     = np.array_equal(cropped_pred, out0)
pixel_correct  = (cropped_pred == out0).sum()
pixel_total    = out0.size
pixel_accuracy = pixel_correct / pixel_total

print(f"\n✅ Grid exact‑match: {grid_match}")
print(f"📊 Pixel accuracy: {pixel_correct}/{pixel_total} = {pixel_accuracy:.2%}")
