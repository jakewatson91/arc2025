import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ────────────────────────────────────────────────────────────────────────────────
#  Hyperparameters / Configuration
# ────────────────────────────────────────────────────────────────────────────────
PATCH_SIZE      = 11             # must be odd
HALF_PATCH      = PATCH_SIZE // 2
NUM_COLORS      = 10             # ARC palette 0–9
BATCH_SIZE      = 64
LEARNING_RATE   = 1e-3
NUM_EPOCHS      = 5
PAD_VALUE       = -1             # will be ignored in loss
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# For positional encoding we only need at most PATCH_SIZE*PATCH_SIZE tokens
MAX_PATCH_SEQUENCE = PATCH_SIZE * PATCH_SIZE

# ────────────────────────────────────────────────────────────────────────────────
#  Utility: convert integer grid → one‑hot tensor (C×H×W)
# ────────────────────────────────────────────────────────────────────────────────
def to_one_hot(grid: np.ndarray, num_classes: int = NUM_COLORS) -> torch.Tensor:
    """
    grid:  H×W numpy array of ints in [0..num_classes-1]
    returns: C×H×W torch.FloatTensor one‑hot encoding
    """
    eye = np.eye(num_classes, dtype=np.float32)    # H×W×C after indexing
    onehot = eye[grid]                             # H×W×C
    onehot = onehot.transpose(2,0,1)               # C×H×W
    return torch.from_numpy(onehot)

# ────────────────────────────────────────────────────────────────────────────────
#  1) Load ARC JSON files
# ────────────────────────────────────────────────────────────────────────────────
with open('data/arc-agi_training_challenges.json')   as f:
    training_challenges = json.load(f)
with open('data/arc-agi_training_solutions.json')    as f:
    training_solutions  = json.load(f)
with open('data/arc-agi_evaluation_challenges.json') as f:
    evaluation_challenges = json.load(f)
with open('data/arc-agi_evaluation_solutions.json') as f:
    evaluation_solutions  = json.load(f)

# ────────────────────────────────────────────────────────────────────────────────
#  2) Extract fixed‑size patches + center labels from the *training* set
#     We accumulate them in Python lists so we can then build a Dataset.
# ────────────────────────────────────────────────────────────────────────────────
patch_tensors = []   # will hold N×(C×P×P) tensors
center_labels = []   # will hold N integers in [0..NUM_COLORS-1]

for task_id, task in training_challenges.items():
    examples = task['train']               # list of dicts: {'input':…, 'output':…}
    solutions = training_solutions[task_id]
    for example_dict, solution_grid in zip(examples, solutions):
        input_grid  = np.array(example_dict['input'])
        output_grid = np.array(solution_grid)

        in_h, in_w = input_grid.shape
        out_h, out_w = output_grid.shape
        canvas_h   = max(in_h, out_h)
        canvas_w   = max(in_w, out_w)

        # pad both to (canvas_h×canvas_w) plus a HALF_PATCH border
        pad_top    = HALF_PATCH
        pad_left   = HALF_PATCH
        pad_bottom = HALF_PATCH + (canvas_h - in_h)
        pad_right  = HALF_PATCH + (canvas_w - in_w)

        padded_input  = np.pad(
            input_grid,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            constant_values=PAD_VALUE
        )
        padded_output = np.pad(
            output_grid,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            constant_values=PAD_VALUE
        )

        # slide a PATCH_SIZE×PATCH_SIZE window
        for row in range(canvas_h):
            for col in range(canvas_w):
                patch_input  = padded_input [row:row+PATCH_SIZE, col:col+PATCH_SIZE]
                patch_output = padded_output[row:row+PATCH_SIZE, col:col+PATCH_SIZE]

                # center of patch is at (HALF_PATCH, HALF_PATCH)
                label = int(patch_output[HALF_PATCH, HALF_PATCH])
                if label == PAD_VALUE:
                    # skip patches whose center is padding
                    continue

                # convert patch_input → one‑hot tensor C×P×P
                patch_tensor = to_one_hot(patch_input)
                patch_tensors.append(patch_tensor)
                center_labels.append(label)

# stack into (N, C, P, P) and (N,)
patch_dataset_tensors = torch.stack(patch_tensors, dim=0)   # (N, C, P, P)
center_label_tensor   = torch.tensor(center_labels, dtype=torch.long)  # (N,)

# ────────────────────────────────────────────────────────────────────────────────
#  3) Dataset & DataLoader
# ────────────────────────────────────────────────────────────────────────────────
class PatchDataset(Dataset):
    def __init__(self, inputs: torch.Tensor, labels: torch.Tensor):
        self.inputs = inputs
        self.labels = labels
    def __len__(self):
        return self.inputs.shape[0]
    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

train_dataset = PatchDataset(patch_dataset_tensors, center_label_tensor)
train_loader  = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

# ────────────────────────────────────────────────────────────────────────────────
#  4) Define Multi‑Scale CNN + Transformer model
# ────────────────────────────────────────────────────────────────────────────────
class MultiScaleTransformerTranslator(nn.Module):
    def __init__(self,
                 num_colors:     int,
                 branch_channels: int,
                 max_patches:    int,
                 num_heads:      int = 4,
                 num_layers:     int = 2):
        super().__init__()
        embed_dim = branch_channels * 3

        # three CNN “views” with different receptive fields
        self.conv3 = nn.Sequential(
            nn.Conv2d(num_colors, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, branch_channels, kernel_size=1)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(num_colors, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv2d(32, branch_channels, kernel_size=1)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(num_colors, 32, kernel_size=7, padding=3), nn.ReLU(),
            nn.Conv2d(32, branch_channels, kernel_size=1)
        )

        # learned positional encoding for sequences up to max_patches
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_patches, embed_dim))

        # small TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # final linear to predict color logits per token
        self.to_logits = nn.Linear(embed_dim, num_colors)

    def forward(self, onehot_patch: torch.Tensor) -> torch.Tensor:
        """
        onehot_patch: (B, C=num_colors, P, P)
        returns: center_logits: (B, num_colors)
        """
        batch_size, C, H, W = onehot_patch.shape

        # 1) multi‑scale CNN featurization
        feat3 = self.conv3(onehot_patch)   # → (B, branch_ch, H, W)
        feat5 = self.conv5(onehot_patch)
        feat7 = self.conv7(onehot_patch)

        # 2) concatenate → (B, embed_dim, H, W)
        features = torch.cat([feat3, feat5, feat7], dim=1)

        # 3) flatten spatial → sequence: (B, H*W, embed_dim)
        sequence = features.view(batch_size, -1, H*W).permute(0,2,1)

        # 4) add positional encoding (truncate if needed)
        sequence = sequence + self.positional_encoding[:, :H*W, :]

        # 5) transformer refinement
        sequence = self.transformer(sequence)   # (B, H*W, embed_dim)

        # 6) project each token → color logits (B, H*W, num_colors)
        sequence = self.to_logits(sequence)

        # 7) extract the *center* token’s logits
        center_index = (H*W) // 2
        center_logits = sequence[:, center_index, :]  # (B, num_colors)
        return center_logits

# ────────────────────────────────────────────────────────────────────────────────
#  5) Instantiate, loss & optimizer, train on patches
# ────────────────────────────────────────────────────────────────────────────────
model     = MultiScaleTransformerTranslator(
                num_colors=NUM_COLORS,
                branch_channels=16,
                max_patches=MAX_PATCH_SEQUENCE,
                num_heads=4,
                num_layers=2
            ).to(DEVICE)
loss_fn   = nn.CrossEntropyLoss()  # center_labels in [0..NUM_COLORS-1]
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(1, NUM_EPOCHS+1):
    model.train()
    running_patch_loss = 0.0
    for batch_patches, batch_centers in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}"):
        batch_patches = batch_patches.to(DEVICE)    # (B, C, P, P)
        batch_centers = batch_centers.to(DEVICE)    # (B,)
        optimizer.zero_grad()
        logits = model(batch_patches)               # (B, NUM_COLORS)
        loss   = loss_fn(logits, batch_centers)
        loss.backward()
        optimizer.step()
        running_patch_loss += loss.item()
    avg_patch_loss = running_patch_loss / len(train_loader)
    print(f"→ Epoch {epoch} avg patch‑classification loss: {avg_patch_loss:.4f}")

# ────────────────────────────────────────────────────────────────────────────────
#  6) Inference on full grids via sliding‑window voting
# ────────────────────────────────────────────────────────────────────────────────
def translate_full_grid(
        trained_model: nn.Module,
        input_grid:    list[list[int]],
        output_shape:  tuple[int,int]
    ) -> list[list[int]]:

    raw_input      = np.array(input_grid)
    in_h, in_w     = raw_input.shape
    out_h, out_w   = output_shape
    canvas_h       = max(in_h, out_h)
    canvas_w       = max(in_w, out_w)

    # pad so we can extract edge patches
    pad_top    = HALF_PATCH
    pad_left   = HALF_PATCH
    pad_bottom = HALF_PATCH + (canvas_h - in_h)
    pad_right  = HALF_PATCH + (canvas_w - in_w)

    padded_input = np.pad(
        raw_input,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        constant_values=PAD_VALUE
    )

    vote_accumulator = torch.zeros((NUM_COLORS, canvas_h, canvas_w), dtype=torch.float32)
    vote_count       = torch.zeros((canvas_h, canvas_w), dtype=torch.float32)

    trained_model.eval()
    with torch.no_grad():
        for row in range(canvas_h):
            for col in range(canvas_w):
                patch = padded_input[row:row+PATCH_SIZE, col:col+PATCH_SIZE]
                onehot_patch = to_one_hot(patch).unsqueeze(0).to(DEVICE)
                center_logits = trained_model(onehot_patch)[0].cpu()  # (NUM_COLORS,)
                vote_accumulator[:, row, col] += center_logits
                vote_count[row, col]         += 1

    average_logits = vote_accumulator / vote_count.unsqueeze(0)  # (NUM_COLORS, canvas_h, canvas_w)
    full_canvas    = average_logits.argmax(dim=0).numpy()        # canvas_h×canvas_w

    # crop back to exactly output_shape
    final_output = full_canvas[
        pad_top: pad_top + out_h,
        pad_left: pad_left + out_w
    ]
    return final_output.tolist()

# ────────────────────────────────────────────────────────────────────────────────
#  7) Evaluate on the *evaluation* set
# ────────────────────────────────────────────────────────────────────────────────
grid_accuracies = []
all_predictions = {}

for task_id, task in evaluation_challenges.items():
    test_input_grid = task['test'][0]['input']
    true_output_grid = np.array(evaluation_solutions[task_id][0])
    predicted_output = translate_full_grid(model, test_input_grid, true_output_grid.shape)
    predicted_array  = np.array(predicted_output)
    accuracy         = (predicted_array == true_output_grid).mean()
    grid_accuracies.append(accuracy == 1.0)
    print(f"Task {task_id}: exact‑grid accuracy = {accuracy:.3f}")
    all_predictions[task_id] = predicted_output

overall_accuracy = sum(grid_accuracies) / len(grid_accuracies)
print(f"\nFinal ARC exact‑grid accuracy: {overall_accuracy:.3f}")

with open('arc_patch_transformer_predictions.json','w') as out_json:
    json.dump(all_predictions, out_json, indent=2)

print("✅ Finished. Predictions saved to arc_patch_transformer_predictions.json")