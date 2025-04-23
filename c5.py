import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path

from models2 import MultiScalePatchTranslator, LLaMAGridTransformer, GridFusionModel, SimpleGridTransformer

# ── Hyperparameters ────────────────────────────────────────────────────────────
PATCH_SIZE    = 11
NUM_COLORS    = 10
BATCH_SIZE    = 16
LEARNING_RATE = 1e-5
EPOCHS        = 100
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
    # evaluation_solutions  = { first_id: evaluation_solutions[first_id]  }
    print(f"[DEBUG] Now using only task {first_id} for both train & eval")

# ── 2) Extract patch pairs & full grids ────────────────────────────────────────
input_patches, output_patches, mask_patches = [], [], []
padded_inputs_np, padded_outputs_np        = [], []

for task_id in training_challenges:
    task      = training_challenges[task_id]
    example   = task['train'][0]
    solution  = training_solutions[task_id][0]
    raw_in    = np.array(example['input'])
    raw_out   = np.array(solution)
    in_h, in_w   = raw_in.shape
    out_h, out_w = raw_out.shape

    canvas_h = max(in_h, out_h)
    canvas_w = max(in_w, out_w)

    # pad input
    pad_top_in    = HALF_PATCH
    pad_bottom_in = HALF_PATCH + (canvas_h - in_h)
    pad_left_in   = HALF_PATCH
    pad_right_in  = HALF_PATCH + (canvas_w - in_w)
    padded_input  = np.pad(raw_in,
                           ((pad_top_in, pad_bottom_in),
                            (pad_left_in, pad_right_in)),
                           constant_values=PAD_VAL)

    # pad output
    pad_top_out    = HALF_PATCH
    pad_bottom_out = HALF_PATCH + (canvas_h - out_h)
    pad_left_out   = HALF_PATCH
    pad_right_out  = HALF_PATCH + (canvas_w - out_w)
    padded_output  = np.pad(raw_out,
                           ((pad_top_out, pad_bottom_out),
                            (pad_left_out, pad_right_out)),
                           constant_values=PAD_VAL)

    padded_inputs_np.append(padded_input)
    padded_outputs_np.append(padded_output)

    print(f"[DEBUG] raw_in {raw_in.shape} → canvas ({canvas_h},{canvas_w}), pads T{pad_top_in},B{pad_bottom_in},L{pad_left_in},R{pad_right_in}")
    print(f"[DEBUG] padded_input shape: {padded_input.shape}")

    # extract patches over output area
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

# ── 3) One‑hot encode / torch tensors ─────────────────────────────────────────
onehot_input_patches = [to_onehot(p).to(DEVICE) for p in input_patches]
output_patch_tensors = [torch.from_numpy(p).long().to(DEVICE) for p in output_patches]
patch_mask_tensor    = [torch.from_numpy(m).float().to(DEVICE) for m in mask_patches]

# full‐grid tensors for transformer
full_grid_inputs  = [to_onehot(g).to(DEVICE) for g in padded_inputs_np]      
full_grid_targets = [torch.from_numpy(g).long().to(DEVICE) for g in padded_outputs_np]

# ── 4) Dataset & DataLoader ───────────────────────────────────────────────────
class PatchTranslatorDataset(Dataset):
    def __init__(self, ins, outs, msks):
        self.ins, self.outs, self.msks = ins, outs, msks
    def __len__(self): return len(self.ins)
    def __getitem__(self, i):
        return self.ins[i], self.outs[i], self.msks[i]

def patch_collate_fn(batch):
    inputs, targets, masks = zip(*batch)
    x = torch.stack(inputs).unsqueeze(1)  # (B,1,C,P,P)
    y = torch.stack(targets).unsqueeze(1) # (B,1,P,P)
    m = torch.stack(masks).unsqueeze(1)   # (B,1,P,P)
    return x, y, m

train_ds     = PatchTranslatorDataset(onehot_input_patches,
                                       output_patch_tensors,
                                       patch_mask_tensor)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True, drop_last=True,
                          collate_fn=patch_collate_fn)
print(f"DataLoader yields {len(train_loader)} batches of size {BATCH_SIZE}")

# ── 5) Model, optimizers, loss ────────────────────────────────────────────────
cnn         = MultiScalePatchTranslator(num_channels=NUM_COLORS).to(DEVICE)
llama       = LLaMAGridTransformer(model_name="facebook/opt-350m",
                                   out_channels=NUM_COLORS).to(DEVICE).half()
transformer = SimpleGridTransformer().to(DEVICE)
fusion      = GridFusionModel(cnn_model=cnn, llama_model=transformer).to(DEVICE)

opt_cnn   = torch.optim.Adam(cnn.parameters(),   lr=LEARNING_RATE)
opt_llama = torch.optim.Adam(llama.parameters(), lr=LEARNING_RATE)
opt_fuse  = torch.optim.Adam(fusion.parameters(),lr=LEARNING_RATE)
loss_fn   = nn.CrossEntropyLoss()
loss_fn_grid = nn.CrossEntropyLoss(ignore_index=PAD_VAL, reduction="none")


# ── 6) Training: patch‑CNN ─────────────────────────────────────────────────────
print("\n🚀 Training Patch CNN")
for epoch in range(1, EPOCHS+1):
    cnn.train()
    total_loss = 0
    for x_patch, y_patch, _ in train_loader:
        x_patch = x_patch.squeeze(1)  # (B,C,P,P)
        y_center= y_patch.squeeze(1)[:,HALF_PATCH,HALF_PATCH]  # (B,)
        logits  = cnn(x_patch)[:, :, HALF_PATCH, HALF_PATCH]   # (B,C)
        loss    = loss_fn(logits, y_center)
        opt_cnn.zero_grad(); loss.backward(); opt_cnn.step()
        total_loss += loss.item()*x_patch.size(0)
    print(f"→ Epoch {epoch} avg patch‑CNN loss: {total_loss/len(train_ds):.5f}")

# ── just after you load the model (before any training) ────────────────────────
# 1) Freeze everything except your proj layer
for name, param in llama.model.named_parameters():
    param.requires_grad = False
print(f"🔒 Frozen {sum(1 for _ in llama.model.parameters())} backbone weights")

# 2) Re‑build your optimizer over only the proj parameters
opt_llama = torch.optim.Adam(
    filter(lambda p: p.requires_grad, llama.parameters()),
    lr=1e-6,                 # ← 10× lower than before
    weight_decay=1e-4        # ← small decay helps stability
)

# 3) Prepare for gradient clipping
max_norm = 1.0


# ── In your training loop ──────────────────────────────────────────────────────
loss_fn_grid = nn.CrossEntropyLoss(ignore_index=PAD_VAL, reduction="none")

print("\n🚀 Training Grid Transformer (frozen backbone + clipped grads)")
for epoch in range(1, EPOCHS+1):
    total_loss = 0.0

    for x_full, y_full in zip(full_grid_inputs, full_grid_targets):
        # build a batch of size 1
        x = x_full.unsqueeze(0).to(DEVICE)  # (1,C,H,W)
        y = y_full.unsqueeze(0).to(DEVICE)  # (1,H,W)

        # forward
        llama.tokenizer.model_max_length = x_full.numel()
        logits = llama(x)                   # (1,C,H,W)

        # flatten
        B,C,H,W     = logits.shape
        flat_logits = logits.permute(0,2,3,1).reshape(-1, C)  # (B*H*W, C)
        flat_labels = y.reshape(-1)                           # (B*H*W,)

        # per‑pixel CE, ignore PAD_VAL
        loss_map = loss_fn_grid(flat_logits, flat_labels)
        valid    = (flat_labels != PAD_VAL)
        loss     = loss_map[valid].mean()

        # backward + clip
        opt_llama.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, llama.parameters()),
            max_norm
        )
        opt_llama.step()

        total_loss += loss.item()

    avg = total_loss / len(full_grid_inputs)
    print(f"→ Epoch {epoch} avg loss: {avg:.5f}")


# ── 8) Training: Fusion Model ─────────────────────────────────────────────────
print("\n🚀 Training Fusion Model")
for epoch in range(1, EPOCHS+1):
    fusion.train()
    epoch_loss = 0
    for b,(x,y,m) in enumerate(tqdm(train_loader, desc=f"Fusion Epoch {epoch}")):
        x, y, m = x.to(DEVICE), y.to(DEVICE).squeeze(1), m.to(DEVICE).squeeze(1)
        if b==0:
            print(" BATCH [x,y,m] shapes:", x.shape, y.shape, m.shape)
        x = x.squeeze(1)
        preds = fusion(x)  # (B,C,P,P)
        if b==0:
            print("  preds shape:", preds.shape,
                  "min/max logits:", preds.min().item(), preds.max().item())
        logits_center  = preds[:,:,HALF_PATCH,HALF_PATCH]
        targets_center = y[:,HALF_PATCH,HALF_PATCH]
        loss = loss_fn(logits_center, targets_center)
        if b==0:
            print("  Center preds:", logits_center.argmax(1).tolist())
            print("  Ground truths:", targets_center.tolist())
        opt_fuse.zero_grad(); loss.backward(); opt_fuse.step()
        epoch_loss += loss.item()*x.size(0)
    print(f"→ Fusion Epoch {epoch} avg loss: {epoch_loss/len(train_ds):.5f}")

# ── 9) Evaluation ──────────────────────────────────────────────────────────────
def translate_single_grid(input_grid, output_shape):
    raw = np.array(input_grid)
    in_h, in_w = raw.shape
    out_h, out_w = output_shape
    canvas_h = max(in_h, out_h)
    canvas_w = max(in_w, out_w)
    pad_top    = HALF_PATCH
    pad_bottom = HALF_PATCH + (canvas_h - in_h)
    pad_left   = HALF_PATCH
    pad_right  = HALF_PATCH + (canvas_w - in_w)

    print(f"\n[DEBUG] raw_in {raw.shape} → canvas ({canvas_h},{canvas_w}), pads T{pad_top},B{pad_bottom},L{pad_left},R{pad_right}")
    padded = np.pad(raw, ((pad_top,pad_bottom),(pad_left,pad_right)), constant_values=PAD_VAL)
    print(f"[DEBUG] padded_input shape: {padded.shape}")

    vote_accum  = torch.zeros((NUM_COLORS, canvas_h, canvas_w), device=DEVICE)
    vote_counts = torch.zeros((canvas_h, canvas_w), device=DEVICE)

    fusion.eval()
    with torch.no_grad():
        for i in range(canvas_h):
            for j in range(canvas_w):
                patch = padded[i:i+PATCH_SIZE, j:j+PATCH_SIZE]
                oh    = to_onehot(patch).unsqueeze(0).to(DEVICE)
                logits= fusion(oh)[0]
                vote_accum[:, i, j]  += logits[:, HALF_PATCH, HALF_PATCH]
                vote_counts[i, j]    += 1

    print(f"[DEBUG] total votes = {vote_counts.sum().item()} (should be {canvas_h*canvas_w})")
    avg_logits  = vote_accum / vote_counts.unsqueeze(0)
    pred_canvas = avg_logits.argmax(dim=0).cpu().numpy()
    print(f"[DEBUG] pre‑crop canvas shape: {pred_canvas.shape}")
    cropped = pred_canvas[:out_h, :out_w]
    print(f"[DEBUG] cropped shape: {cropped.shape}")
    if cropped.shape != (out_h, out_w):
        print("!!! Shape mismatch !!!")
    return cropped.tolist()

def evaluate_metrics(tasks, sols, name=""):
    grid_match = pixel_match = pixel_total = 0
    print(f"\n──── 🔍 {name} Evaluation ────")
    for tid in tasks:
        inp = tasks[tid]['test'][0]['input']
        tgt = np.array(sols[tid][0])
        print(f"\n📌 Task {tid}")
        print("Input grid:\n", inp)
        print("Target grid:\n", tgt)

        pred = np.array(translate_single_grid(inp, tgt.shape))
        print("Pred grid:\n", pred)

        match = np.array_equal(pred, tgt)
        acc   = np.mean(pred == tgt)

        print(f"✓ Grid match: {match}")
        print(f"🎯 Pixel accuracy: {acc:.2%}")
        if not match:
            diffs = np.argwhere(pred != tgt)
            print(f"✗ Mismatches: {len(diffs)} pixels:\n", diffs[:5])

        grid_match  += int(match)
        pixel_match += (pred == tgt).sum()
        pixel_total += tgt.size

    print(f"\n📊 {name} Results: {grid_match}/{len(tasks)} exact matches")
    print(f"🧮 Total pixel acc: {pixel_match}/{pixel_total} = {pixel_match/pixel_total:.2%}")

# Final evaluation on training task
evaluate_metrics(training_challenges, training_solutions, name="Training (1 Task)")
