import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import trange

# ── Hyperparameters ───────────────────────────────────────────
NUM_COLORS    = 10                   # ARC uses colors 0–9
PAD_VAL       = NUM_COLORS           # we’ll treat “10” as our padding index
IN_CHANS      = NUM_COLORS + 1       # input one‐hot channels (0–9 + PAD)
NUM_CLASSES   = NUM_COLORS + 1       # same for output logits
LR            = 1e-3
EPOCHS        = 10
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── U‑Net Definition ──────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.enc1 = ConvBlock(in_ch,   32)
        self.enc2 = ConvBlock(  32,   64)
        self.enc3 = ConvBlock(  64,  128)
        self.up3  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, output_padding=1)
        self.dec2 = ConvBlock(64+64, 64)
        self.up2  = nn.ConvTranspose2d( 64, 32, kernel_size=2, stride=2, output_padding=1)
        self.dec1 = ConvBlock(32+32, 32)
        self.outc = nn.Conv2d(32, num_classes, kernel_size=1)
    def forward(self, x):
        e1 = self.enc1(x)        # → (B,32,H,W)
        p1 = F.max_pool2d(e1, 2) # → (B,32,H/2,W/2)
        e2 = self.enc2(p1)       # → (B,64,H/2,W/2)
        p2 = F.max_pool2d(e2, 2) # → (B,64,H/4,W/4)
        e3 = self.enc3(p2)       # → (B,128,H/4,W/4)
        u3 = self.up3(e3)        # → (B,64,H/2,W/2)
        d2 = self.dec2(torch.cat([u3,e2],dim=1))
        u2 = self.up2(d2)        # → (B,32,H,W)
        d1 = self.dec1(torch.cat([u2,e1],dim=1))
        return self.outc(d1)     # → (B,num_classes,H,W)

model = UNet(IN_CHANS, NUM_CLASSES).to(DEVICE)
opt   = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_VAL)

# ── Helpers ────────────────────────────────────────────────────
def onehot_grid(grid: np.ndarray) -> torch.Tensor:
    """Convert H×W int grid (0..10) → (C,H,W) one‑hot tensor."""
    C = NUM_COLORS+1
    oh = np.eye(C, dtype=np.float32)[grid]    # H×W×C
    return torch.from_numpy(oh).permute(2,0,1)  # C×H×W

def pad_to_canvas(inp, out):
    """
    Given two 2D arrays inp, out, pad both up to the same H×W,
    filling with PAD_VAL.
    Returns padded_in, padded_out.
    """
    h_in, w_in = inp.shape
    h_out, w_out = out.shape
    H = max(h_in, h_out)
    W = max(w_in, w_out)
    # center‐anchor both into H×W
    pad_i = ((0, H-h_in), (0, W-w_in))
    pad_o = ((0, H-h_out), (0, W-w_out))
    pi = np.pad(inp, pad_i, constant_values=PAD_VAL)
    po = np.pad(out, pad_o, constant_values=PAD_VAL)
    return pi, po

# ── Load ARC JSON ───────────────────────────────────────────────
base = Path("data")
train_ch     = json.loads((base/"arc-agi_training_challenges.json").read_text())
train_sol    = json.loads((base/"arc-agi_training_solutions.json").read_text())
eval_ch      = json.loads((base/"arc-agi_evaluation_challenges.json").read_text())
eval_sol     = json.loads((base/"arc-agi_evaluation_solutions.json").read_text())

# ── 1) Training Loop ───────────────────────────────────────────
for epoch in range(1, EPOCHS+1):
    model.train()
    total_loss = 0.0
    n_steps    = 0
    for tid, task in train_ch.items():
        for ex, sol in zip(task["train"], train_sol[tid]):
            A = np.array(ex["input"], dtype=int)
            B = np.array(sol,   dtype=int)
            Ai, Bi = pad_to_canvas(A, B)
            x = onehot_grid(Ai).unsqueeze(0).to(DEVICE)   # (1,C,H,W)
            y = torch.from_numpy(Bi).long().unsqueeze(0).to(DEVICE)  # (1,H,W)
            logits = model(x)                             # (1,classes,H,W)
            loss   = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_steps    += 1
    print(f"Epoch {epoch}/{EPOCHS}  avg loss: {total_loss/n_steps:.4f}")

# ── 2) Inference & Save ──────────────────────────────────────────
preds = {}
model.eval()
with torch.no_grad():
    for tid, task in eval_ch.items():
        A = np.array(task["test"][0]["input"], dtype=int)
        B = np.array(eval_sol[tid][0], dtype=int)
        Ai, _ = pad_to_canvas(A, B)
        x = onehot_grid(Ai).unsqueeze(0).to(DEVICE)
        logits = model(x)[0]             # (classes,H,W)
        canvas_pred = logits.argmax(0).cpu().numpy()
        # crop to target shape
        Ht, Wt = B.shape
        P = canvas_pred[:Ht, :Wt].tolist()
        preds[tid] = P

# write predictions
(Path.cwd()/"arc_unet_preds.json").write_text(json.dumps(preds, indent=2))
print("Done → arc_unet_preds.json")