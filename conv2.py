#!/usr/bin/env python3
"""
arc_cnn_baseline.py

A minimal fully‐convolutional ARC solver:
 1) pads every grid to a fixed canvas size,
 2) trains a simple CNN to predict per‐cell colors,
 3) ignores padded cells in the loss,
 4) crops back to original size at inference.

Usage:
    python arc_cnn_baseline.py
"""

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ── Configuration ────────────────────────────────────────────────────────────────

TRAINING_CHALLENGES_PATH      = "data/arc-agi_training_challenges.json"
TRAINING_SOLUTIONS_PATH       = "data/arc-agi_training_solutions.json"
EVAL_CHALLENGES_PATH          = "data/arc-agi_evaluation_challenges.json"
EVAL_SOLUTIONS_PATH           = "data/arc-agi_evaluation_solutions.json"
OUTPUT_PREDICTIONS_PATH       = "arc_cnn_baseline_predictions.json"

# ── Hyperparameters ────────────────────────────────────────────────────────────
PATCH_SIZE    = 11           # must be odd
HALF_PATCH    = PATCH_SIZE // 2
NUM_COLORS    = 10           # ARC uses colors 0–9
EMBED_DIM     = 64           # per‐token embedding size
TRANSFORMER_HEADS = 4
TRANSFORMER_LAYERS = 2
BATCH_SIZE    = 128
LEARNING_RATE = 1e-3
EPOCHS        = 1
PAD_VALUE     = -1           # we'll pad with a special value
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Utility: one‑hot encoding of an H×W integer grid → C×H×W tensor ────────────
def grid_to_onehot(grid: np.ndarray, num_classes: int=NUM_COLORS) -> torch.Tensor:
    eye = np.eye(num_classes, dtype=np.float32)      # shape (num_classes,num_classes)
    onehot = eye[grid.clip(0, num_classes-1)]        # H×W×C, padded entries will be clipped
    return torch.from_numpy(onehot.transpose(2,0,1)) # C×H×W

# ── 1) Load ARC train data ─────────────────────────────────────────────────────
with open('data/arc-agi_training_challenges.json') as f:
    train_challenges = json.load(f)
with open('data/arc-agi_training_solutions.json') as f:
    train_solutions  = json.load(f)

# ── 2) Extract fixed‐size patches + center labels ──────────────────────────────
patch_tensors = []
center_labels = []

for task_id, task in train_challenges.items():
    examples = task['train']
    solutions = train_solutions[task_id]
    for example, solution in zip(examples, solutions):
        input_grid  = np.array(example['input'])
        output_grid = np.array(solution)

        H_in, W_in = input_grid.shape
        H_out, W_out = output_grid.shape
        # pad both to the same “canvas” size so every patch exists
        canvas_h = max(H_in, H_out)
        canvas_w = max(W_in, W_out)
        pad_top    = HALF_PATCH
        pad_left   = HALF_PATCH
        pad_bottom = HALF_PATCH + (canvas_h - H_in)
        pad_right  = HALF_PATCH + (canvas_w - W_in)
        padded_input = np.pad(input_grid,
                              ((pad_top, pad_bottom),
                               (pad_left, pad_right)),
                              constant_values=PAD_VALUE)
        pad_bottom_out = HALF_PATCH + (canvas_h - H_out)
        pad_right_out  = HALF_PATCH + (canvas_w - W_out)
        padded_output = np.pad(output_grid,
                               ((pad_top, pad_bottom_out),
                                (pad_left, pad_right_out)),
                               constant_values=PAD_VALUE)

        # slide a PATCH_SIZE×PATCH_SIZE window over the input canvas
        for i in range(canvas_h):
            for j in range(canvas_w):
                patch = padded_input[i : i+PATCH_SIZE, j : j+PATCH_SIZE]
                # label = center pixel of the *output* patch
                label = padded_output[i + HALF_PATCH, j + HALF_PATCH]
                # skip if that center is also padding
                if label == PAD_VALUE:
                    continue

                # convert patch → one‑hot tensor C×P×P
                patch_tensor = grid_to_onehot(patch)
                patch_tensors.append(patch_tensor)
                center_labels.append(int(label))

# stack into tensors
patch_dataset = torch.stack(patch_tensors, dim=0)   # (N, C, P, P)
label_tensor  = torch.tensor(center_labels, dtype=torch.long)  # (N,)

# ── 3) PyTorch Dataset & DataLoader ───────────────────────────────────────────
class PatchDataset(Dataset):
    def __init__(self, patches, labels):
        self.patches = patches
        self.labels  = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.patches[idx], self.labels[idx]

train_loader = DataLoader(
    PatchDataset(patch_dataset, label_tensor),
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

# ── 4) Define CNN+Transformer model ────────────────────────────────────────────
class CNNTransformerPatchClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # a small 2‑layer CNN to map C×P×P → EMBED_DIM×P×P
        self.cnn = nn.Sequential(
            nn.Conv2d(NUM_COLORS, EMBED_DIM, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(EMBED_DIM, EMBED_DIM, kernel_size=1),
            nn.ReLU(),
        )
        # positional encoding for P*P tokens
        num_tokens = PATCH_SIZE * PATCH_SIZE
        self.positional_encoding = nn.Parameter(torch.randn(1, num_tokens, EMBED_DIM))
        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=TRANSFORMER_HEADS,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, TRANSFORMER_LAYERS)
        # classifier from center‐token embedding to NUM_COLORS logits
        self.classifier = nn.Linear(EMBED_DIM, NUM_COLORS)

    def forward(self, x):
        # x is (B, C, P, P)
        batch_size = x.shape[0]
        # 1) CNN → (B, EMBED_DIM, P, P)
        features = self.cnn(x)
        # 2) flatten spatial → sequence of length P*P
        B, E, H, W = features.shape
        seq = features.view(B, E, H*W).permute(0, 2, 1)      # (B, P*P, EMBED_DIM)
        # 3) add positional encoding
        seq = seq + self.positional_encoding[:, :H*W, :]
        # 4) transformer refinement
        seq = self.transformer(seq)                         # (B, P*P, EMBED_DIM)
        # 5) pick out the *center* token at index center_idx
        center_idx = (PATCH_SIZE//2)*PATCH_SIZE + (PATCH_SIZE//2)
        center_repr = seq[:, center_idx, :]                 # (B, EMBED_DIM)
        # 6) classification head
        logits = self.classifier(center_repr)               # (B, NUM_COLORS)
        return logits

model = CNNTransformerPatchClassifier().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn   = nn.CrossEntropyLoss()

# ── 5) Training loop ───────────────────────────────────────────────────────────
for epoch in range(1, EPOCHS+1):
    model.train()
    running_loss = 0.0
    for batch_patches, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
        batch_patches = batch_patches.to(DEVICE)
        batch_labels  = batch_labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(batch_patches)
        loss   = loss_fn(logits, batch_labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_patches.size(0)

    avg_loss = running_loss / len(train_loader.dataset)
    print(f"→ Epoch {epoch} average loss: {avg_loss:.4f}")

# ── 5) Inference on evaluation set ──────────────────────────────────────────────

with open(EVAL_CHALLENGES_PATH) as cf, open(EVAL_SOLUTIONS_PATH) as sf:
    eval_challenges = json.load(cf)
    eval_solutions  = json.load(sf)

def predict_grid(input_grid: list[list[int]]) -> list[list[int]]:
    """
    Pad a single input to (max_height, max_width),
    run the CNN, crop back to original size.
    """
    h, w = len(input_grid), len(input_grid[0])
    arr = np.array(input_grid, dtype=int)

    pad_h = canvas_h - h
    pad_w = canvas_w  - w

    padded_arr = np.pad(arr, ((0,pad_h),(0,pad_w)), constant_values=0)
    one_hot   = np.eye(NUM_COLORS, dtype=np.float32)[padded_arr]
    one_hot   = one_hot.transpose(2,0,1)[None]  # shape (1, C, Hmax, Wmax)

    tensor_in = torch.from_numpy(one_hot).to(DEVICE)
    model.eval()
    with torch.no_grad():
        logits = model(tensor_in)[0]              # (C, Hmax, Wmax)
        predicted_full = logits.argmax(dim=0).cpu().numpy()

    # crop back
    return predicted_full[:h, :w].tolist()


all_predictions = {}
total_tasks = 0
correct_tasks = 0

for task_id, task in eval_challenges.items():
    test_input = task["test"][0]["input"]
    ground_truth = np.array(eval_solutions[task_id][0], dtype=int)

    predicted = predict_grid(test_input)
    all_predictions[task_id] = predicted

    # grid‐level correctness
    predicted_arr = np.array(predicted, dtype=int)
    is_exact_match = np.array_equal(predicted_arr, ground_truth)
    correct_tasks += int(is_exact_match)
    total_tasks   += 1

print(f"Evaluation: {correct_tasks}/{total_tasks} tasks correct "
      f"→ {100*correct_tasks/total_tasks:.1f}% grid accuracy")

# save JSON
with open(OUTPUT_PREDICTIONS_PATH, "w") as out_f:
    json.dump(all_predictions, out_f, indent=2)

print(f"Predictions written to {OUTPUT_PREDICTIONS_PATH}")