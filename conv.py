import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models2 import MultiScaleTransformerTranslator

# ── Hyperparameters ────────────────────────────────────────────────────────────
PATCH_SIZE    = 11          # Size of each square patch (must be odd) -- seems 11 is max possible
NUM_COLORS    = 10           # ARC uses colors 0–9
BATCH_SIZE    = 16
LEARNING_RATE = 1e-3
EPOCHS        = 5
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HALF_PATCH    = PATCH_SIZE // 2
PAD_VAL = 0

# ── One‑hot encoding helper ────────────────────────────────────────────────────
def to_onehot(grid: np.ndarray, num_classes: int = NUM_COLORS) -> torch.Tensor:
    """
    Convert an H×W integer grid into a one‑hot tensor of shape C×H×W.
    """
    eye = np.eye(num_classes, dtype=np.float32)      # H×W×C after indexing
    onehot = eye[grid]                               # H×W×C
    return torch.from_numpy(onehot.transpose(2, 0, 1))  # C×H×W

# ── 1) Load ARC data ───────────────────────────────────────────────────────────
with open('data/arc-agi_training_challenges.json') as f:
    training_challenges = json.load(f)
    # print(training_challenges['ff805c23']['test'])
with open('data/arc-agi_training_solutions.json') as f:
    training_solutions = json.load(f)
    # print(training_solutions['00576224'][0])
with open('data/arc-agi_evaluation_challenges.json') as f:
    evaluation_challenges = json.load(f)
    # print(evaluation_challenges['0934a4d8'])
with open('data/arc-agi_evaluation_solutions.json') as f:
    evaluation_solutions = json.load(f)

# ── 2) Extract patch pairs from training examples ──────────────────────────────
input_patches  = []
output_patches = []
mask_patches = [] # don't consider padding in loss

for task_id, task in training_challenges.items():
    # print(task_id)
    # print(task)
    train_examples = task['train']
    test_example = task['test']
    solutions = training_solutions[task_id]
    for example, solution in zip(train_examples, solutions):
        raw_input  = np.array(example['input'])
        raw_output = np.array(example['output'])
        # raw_output = np.array(solution) # for testing
        in_h, in_w = raw_input.shape
        out_h, out_w = raw_output.shape

        # Determine canvas size = max of input/output
        canvas_h = max(in_h, out_h)
        canvas_w = max(in_w, out_w)

        # Pad input up to (canvas_h + 2*HALF_PATCH, canvas_w + 2*HALF_PATCH)
        pad_top_in    = HALF_PATCH
        pad_bottom_in = HALF_PATCH + (canvas_h - in_h)
        pad_left_in   = HALF_PATCH
        pad_right_in  = HALF_PATCH + (canvas_w - in_w)
        padded_input = np.pad(
            raw_input,
            ((pad_top_in, pad_bottom_in), (pad_left_in, pad_right_in)),
            constant_values=PAD_VAL
        )
        # print("Padded Input: ", padded_input)

        # Pad output the same way
        pad_top_out    = HALF_PATCH
        pad_bottom_out = HALF_PATCH + (canvas_h - out_h)
        pad_left_out   = HALF_PATCH
        pad_right_out  = HALF_PATCH + (canvas_w - out_w)
        padded_output = np.pad(
            raw_output,
            ((pad_top_out, pad_bottom_out), (pad_left_out, pad_right_out)),
            constant_values=PAD_VAL
        )
        # print("Padded Output: ", padded_output)

        # Slide a PATCH_SIZE×PATCH_SIZE window over the canvas
        for row_idx in range(canvas_h):
            for col_idx in range(canvas_w):
                patch_in  = padded_input[row_idx : row_idx + PATCH_SIZE,
                                         col_idx : col_idx + PATCH_SIZE]
                patch_out = padded_output[row_idx : row_idx + PATCH_SIZE,
                                          col_idx : col_idx + PATCH_SIZE]
                patch_mask = (patch_out != PAD_VAL).astype(np.float32)

                input_patches.append(patch_in)
                output_patches.append(patch_out)
                mask_patches.append(patch_mask)
    #         break
    #     break    
    # break

# One‑hot encode
onehot_input_patches  = [to_onehot(p) for p in input_patches]
onehot_output_patches = [to_onehot(p) for p in output_patches]
patch_mask_tensor = [torch.from_numpy(m) for m in mask_patches]

# ── 3) Dataset & DataLoader ────────────────────────────────────────────────────
class PatchTranslatorDataset(Dataset):
    def __init__(self, inputs, targets, masks):
        self.inputs  = inputs
        self.targets = targets
        self.masks = masks
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx], self.masks[idx]

training_dataset = PatchTranslatorDataset(onehot_input_patches,
                                          onehot_output_patches,
                                          patch_mask_tensor)
training_loader  = DataLoader(training_dataset,
                              batch_size=min(BATCH_SIZE, len(training_dataset)),
                              shuffle=True,
                              drop_last=True)

model       = MultiScaleTransformerTranslator(num_colors=NUM_COLORS, max_patches=900, branch_ch=16).to(DEVICE)
optimizer   = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn     = nn.CrossEntropyLoss(reduction="none")

# ── 5) Training loop ───────────────────────────────────────────────────────────
for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for batch_inputs, batch_targets, batch_masks in tqdm(training_loader,
                                            desc=f"Epoch {epoch}/{EPOCHS}"):
        batch_inputs  = batch_inputs.to(DEVICE)
        batch_targets = batch_targets.to(DEVICE)
        batch_masks = batch_masks.to(DEVICE)

        predictions   = model(batch_inputs)
        loss          = loss_fn(predictions, batch_targets)
        masks = batch_masks.unsqueeze(1) # add C dim

        loss_masked = loss * masks # loss * 0 for padded cells
        real_loss = loss_masked.mean()
    
        optimizer.zero_grad()
        real_loss.backward()
        optimizer.step()
        epoch_loss += real_loss.item() * batch_inputs.size(0)
    avg_loss = epoch_loss / len(training_dataset)
    # print(batch_inputs.shape)
    print(f"→ Epoch {epoch} avg loss: {avg_loss:.5f}")

# ── 6) Inference / translate a new grid ────────────────────────────────────────
def translate_single_grid(input_grid: list[list[int]],
                          output_shape: tuple[int,int]) -> list[list[int]]:
    """
    Pads the input up to the max(input,output) canvas,
    runs the CNN on every center pixel of that canvas,
    then crops out exactly the output_shape region.
    """
    raw_in = np.array(input_grid)
    in_h, in_w = raw_in.shape
    out_h, out_w = output_shape

    # 1) figure out the canvas
    canvas_h = max(in_h, out_h)
    canvas_w = max(in_w, out_w)

    # 2) same padding on all sides so we can slide patches
    pad_top    = HALF_PATCH
    pad_left   = HALF_PATCH
    pad_bottom = HALF_PATCH + (canvas_h - in_h)
    pad_right  = HALF_PATCH + (canvas_w - in_w)

    padded_in = np.pad(raw_in,
                       ((pad_top, pad_bottom),
                        (pad_left, pad_right)),
                       constant_values=PAD_VAL)

    # 3) accumulate votes over the entire canvas
    vote_accum  = torch.zeros((NUM_COLORS, canvas_h, canvas_w))
    vote_counts = torch.zeros((canvas_h, canvas_w))

    model.eval()
    with torch.no_grad():
        for i in range(canvas_h):
            for j in range(canvas_w):
                patch = padded_in[i : i+PATCH_SIZE, j : j+PATCH_SIZE]
                oh    = to_onehot(patch).unsqueeze(0).to(DEVICE)  # 1×C×P×P
                logits_patch = model(oh)[0].cpu()                # C×P×P
                # take the center pixel’s logits
                vote_accum[:, i, j]  += logits_patch[:, HALF_PATCH, HALF_PATCH]
                vote_counts[i, j]    += 1

    # 4) normalize and pick the highest‐score color
    avg_logits      = vote_accum / vote_counts.unsqueeze(0)
    predicted_canvas = avg_logits.argmax(dim=0).numpy()   # canvas_h×canvas_w

    # 5) now crop out the exact output window
    cropped = predicted_canvas[
        pad_top: pad_top + out_h,
        pad_left: pad_left + out_w
    ]

    return cropped.tolist()

# ── 7) Run on all evaluation tasks & save results ──────────────────────────────
all_predictions = {}
matches = []
for task_id, task in evaluation_challenges.items():
    # print("Task: ", task_id)
    test_example = task['test'][0]['input']
    solution = np.array(evaluation_solutions[task_id][0])
    # print(f"Test Example: {test_example}")
    # print("Sol shape: ", solution.shape)
    all_predictions[task_id] = translate_single_grid(test_example, solution.shape)
    test_pred = np.array(all_predictions[task_id])
    # print(f"Test pred: {all_predictions[task_id]}\n")
    # print("Solution: ", solution)
    accuracy = (test_pred == solution).mean()
    print(f"Accuracy: {accuracy:.2f}\n------------")
    matches.append(int(accuracy) == 1)
final_accuracy = sum(matches) / len(matches)
print(f"Final Accuracy: {final_accuracy}")
    # break

with open('arc_patch_translator_full_named_preds.json', 'w') as out_file:
    json.dump(all_predictions, out_file, indent=2)

print("✅ Finished! Predictions written to arc_patch_translator_full_named_preds.json")