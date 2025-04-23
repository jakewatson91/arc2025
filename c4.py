import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path

from models2 import PatchSeqTransformer

# ── Hyperparameters ────────────────────────────────────────────────────────────
PATCH_SIZE    = 11
NUM_COLORS    = 10
BATCH_SIZE    = 16
LEARNING_RATE = 1e-5
EPOCHS        = 500
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HALF_PATCH    = PATCH_SIZE // 2
PAD_VAL       = 0

# ── One‑hot encoding helper ────────────────────────────────────────────────────
def to_onehot(grid: np.ndarray, num_classes: int = NUM_COLORS) -> torch.Tensor:
    eye    = np.eye(num_classes, dtype=np.float32)
    onehot = eye[grid]
    return torch.from_numpy(onehot.transpose(2,0,1))

# ── 1) Load ARC data ───────────────────────────────────────────────────────────
data_dir = Path("data")
with open(data_dir/"arc-agi_training_challenges.json") as f:
    training_challenges = json.load(f)
with open(data_dir/"arc-agi_training_solutions.json") as f:
    training_solutions = json.load(f)
with open(data_dir/"arc-agi_evaluation_challenges.json") as f:
    evaluation_challenges = json.load(f)
with open(data_dir/"arc-agi_evaluation_solutions.json") as f:
    evaluation_solutions = json.load(f)

print(f"Loaded {len(training_challenges)} training tasks, "
      f"{len(evaluation_challenges)} eval tasks")

# ── DEBUG: restrict to a single task/example ─────────────────────────────────
DEBUG_SINGLE = True
if DEBUG_SINGLE:
    first_id = next(iter(training_challenges))
    training_challenges = { first_id: training_challenges[first_id] }
    training_solutions  = { first_id: training_solutions[first_id]   }
    # evaluation_challenges = { first_id: evaluation_challenges[first_id] }
    # evaluation_solutions  = { first_id: evaluation_solutions[first_id]   }
    print(f"[DEBUG] Now using only task {first_id} for both train & eval")

# Decide which IDs to loop over
train_task_ids = list(training_challenges.keys())
eval_task_ids  = list(evaluation_challenges.keys())

# ── 2) Extract patch pairs ────────────────────────────────────────────────────
input_patches, output_patches, mask_patches = [], [], []

for task_id in train_task_ids:
    task     = training_challenges[task_id]
    examples = task['train']
    solutions= training_solutions[task_id]

    # in DEBUG_SINGLE mode, examples & solutions will each be length 1
    for example, solution in zip(examples, solutions):
        raw_input  = np.array(example['input'])
        raw_output = np.array(solution)
        in_h, in_w   = raw_input.shape
        out_h, out_w = raw_output.shape

        canvas_h = max(in_h, out_h)
        canvas_w = max(in_w, out_w)

        # pad input
        pad_top_in    = HALF_PATCH
        pad_bottom_in = HALF_PATCH + (canvas_h - in_h)
        pad_left_in   = HALF_PATCH
        pad_right_in  = HALF_PATCH + (canvas_w - in_w)
        padded_input = np.pad(
            raw_input,
            ((pad_top_in, pad_bottom_in),
             (pad_left_in, pad_right_in)),
            constant_values=PAD_VAL
        )

        # pad output
        pad_top_out    = HALF_PATCH
        pad_bottom_out = HALF_PATCH + (canvas_h - out_h)
        pad_left_out   = HALF_PATCH
        pad_right_out  = HALF_PATCH + (canvas_w - out_w)
        padded_output = np.pad(
            raw_output,
            ((pad_top_out, pad_bottom_out),
             (pad_left_out, pad_right_out)),
            constant_values=PAD_VAL
        )

        # now actually append into your lists
        for row in range(out_h):
            for col in range(out_w):
                patch_in   = padded_input[row:row+PATCH_SIZE,
                                          col:col+PATCH_SIZE]
                patch_out  = padded_output[row:row+PATCH_SIZE,
                                           col:col+PATCH_SIZE]
                patch_mask = (patch_out != PAD_VAL).astype(np.float32)

                input_patches.append(patch_in)
                output_patches.append(patch_out)
                mask_patches.append(patch_mask)

print(f"Total patches extracted: {len(input_patches)}")

# one‑hot encode / torch tensors
onehot_input_patches = [to_onehot(p).to(DEVICE) for p in input_patches]
output_patch_tensors = [torch.from_numpy(p).long().to(DEVICE) for p in output_patches]
patch_mask_tensor    = [torch.from_numpy(m).float().to(DEVICE) for m in mask_patches]

def patch_collate_fn(batch):
    inputs, targets, masks = zip(*batch)
    x = torch.stack(inputs).unsqueeze(1)
    y = torch.stack(targets).unsqueeze(1)
    m = torch.stack(masks).unsqueeze(1)
    return x, y, m

# ── 3) Dataset & DataLoader ────────────────────────────────────────────────────
class PatchTranslatorDataset(Dataset):
    def __init__(self, ins, outs, msks):
        self.ins, self.outs, self.msks = ins, outs, msks
    def __len__(self): return len(self.ins)
    def __getitem__(self, i): return self.ins[i], self.outs[i], self.msks[i]

train_ds = PatchTranslatorDataset(onehot_input_patches,
                                  output_patch_tensors,
                                  patch_mask_tensor)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          drop_last=True, collate_fn=patch_collate_fn)

print(f"DataLoader yields {len(train_loader)} batches of size {BATCH_SIZE}")

# ── 4) Model, optimizer, loss ───────────────────────────────────────────────────
model     = PatchSeqTransformer(NUM_COLORS, max_patches=900, branch_ch=16).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn   = nn.CrossEntropyLoss(reduction="mean")

# ── 5) Training loop with debug ────────────────────────────────────────────────
for epoch in range(1, EPOCHS+1):
    model.train()
    epoch_loss = 0
    for b, (x,y,m) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        x, y, m = x.to(DEVICE), y.to(DEVICE).squeeze(1), m.to(DEVICE).squeeze(1)
        # debug batch shapes
        if b==0:
            print(" BATCH [x,y,m] shapes:", x.shape, y.shape, m.shape)
        preds = model(x)
        # debug preds
        if b==0:
            print("  preds shape:", preds.shape,
                  " min/max logits:", preds.min().item(), preds.max().item())
        center = PATCH_SIZE // 2
        logits_center = preds[:, :, center, center]         # shape (B, num_classes)
        targets_center = y[:, center, center]               # shape (B,)
        loss = loss_fn(logits_center, targets_center)       # scalar

        if b == 0:
            preds_labels = logits_center.argmax(dim=1)
            print(f"  Center predictions: {preds_labels.tolist()}")
            print(f"  Ground truths     : {targets_center.tolist()}")
            print(f"  loss: {loss.item():.5f}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()*x.size(0)
    print(f"→ Epoch {epoch} avg loss: {epoch_loss/len(train_ds):.5f}")

def translate_single_grid(input_grid: list[list[int]],
                          output_shape: tuple[int,int]) -> list[list[int]]:
    raw_in = np.array(input_grid)
    in_h, in_w = raw_in.shape
    out_h, out_w = output_shape

    # 1) compute canvas + pads
    canvas_h = max(in_h, out_h)
    canvas_w = max(in_w, out_w)
    pad_top    = HALF_PATCH
    pad_left   = HALF_PATCH
    pad_bottom = HALF_PATCH + (canvas_h - in_h)
    pad_right  = HALF_PATCH + (canvas_w - in_w)
    print(f"\n[DEBUG] raw_in {raw_in.shape} → canvas ({canvas_h},{canvas_w}), pads T{pad_top},B{pad_bottom},L{pad_left},R{pad_right}")

    # 2) pad input
    padded_input = np.pad(
        raw_in,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        constant_values=PAD_VAL
    )
    print(f"[DEBUG] padded_input shape: {padded_input.shape}")

    # 3) vote accumulators
    vote_accum  = torch.zeros((NUM_COLORS, canvas_h, canvas_w), device=DEVICE)
    vote_counts = torch.zeros((canvas_h, canvas_w), device=DEVICE)

    # 4) slide & collect center‑logits
    model.eval()
    with torch.no_grad():
        for i in range(canvas_h):
            for j in range(canvas_w):
                patch = padded_input[i : i+PATCH_SIZE, j : j+PATCH_SIZE]
                oh    = to_onehot(patch).unsqueeze(0).to(DEVICE)
                logits_patch = model(oh)[0]
                vote_accum[:, i, j]  += logits_patch[:, HALF_PATCH, HALF_PATCH]
                vote_counts[i, j]    += 1

    print(f"[DEBUG] total votes = {vote_counts.sum().item()} (should be {canvas_h*canvas_w})")

    # 5) aggregate & pick colors
    avg_logits       = vote_accum / vote_counts.unsqueeze(0)
    predicted_canvas = avg_logits.argmax(dim=0).cpu().numpy()
    print(f"[DEBUG] pre‑crop canvas shape: {predicted_canvas.shape}")

    # 6) crop the top‑left out_h×out_w of the canvas
    print(f"[DEBUG] cropping to rows 0:{out_h}, cols 0:{out_w}")
    cropped = predicted_canvas[:out_h, :out_w]

    print(f"[DEBUG] cropped shape: {cropped.shape}")
    if cropped.shape != (out_h, out_w):
        print("!!! Shape mismatch !!!")
    return cropped.tolist()


def evaluate_metrics(tasks, sols, name=""):
    grid_match, pixel_match, pixel_total = 0, 0, 0
    print(f"\n──── 🔍 {name} Evaluation ────")
    for tid in tasks:
        task = tasks[tid]

        input_grid = task['test'][0]['input']
        print(f"Input grid: {input_grid}\n")
        target_grid = np.array(sols[tid][0])
        print(f"Target grid: {target_grid}\n")
        pred_grid = np.array(translate_single_grid(input_grid, target_grid.shape))
        print(f"Pred grid: {pred_grid}\n")

        is_match = np.array_equal(pred_grid, target_grid)
        pixel_acc = np.mean(pred_grid == target_grid)

        print(f"\n📌 Task {tid}")
        print(f"✓ Grid match: {is_match}")
        print(f"🎯 Pixel accuracy: {pixel_acc:.2%}")
        if not is_match:
            diffs = np.argwhere(pred_grid != target_grid)
            print(f"✗ Mismatches: {len(diffs)} pixels:\n", diffs)
            print("Target sample:\n", target_grid)
            print("Pred sample:\n", pred_grid)

        grid_match += int(is_match)
        pixel_match += np.sum(pred_grid == target_grid)
        pixel_total += target_grid.size

    print(f"\n📊 {name} Results: {grid_match} / {len(tasks)} exact matches")
    print(f"🧮 Total pixel acc: {pixel_match}/{pixel_total} = {pixel_match/pixel_total:.2%}")

# evaluate_metrics(training_challenges, training_solutions, name="Training")
# evaluate_metrics(evaluation_challenges, evaluation_solutions, name="Eval")
evaluate_metrics(
    { "00576224": training_challenges["00576224"] },
    { "00576224": training_solutions["00576224"] },
    name="Training (1 Task)"
)
