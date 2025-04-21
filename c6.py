import json
import numpy as np
import torch
import torch.nn as nn

# ── Hyperparameters ────────────────────────────────────────────────────────────
PATCH_SIZE    = 11
GRID_DIM      = 30
NUM_COLORS    = 10
LR            = 1e-5
EPOCHS        = 500
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HALF_PATCH    = PATCH_SIZE // 2
PAD_VAL       = -1

# ── One‑hot encoding helper ────────────────────────────────────────────────────
def to_onehot(grid: np.ndarray, num_classes: int = NUM_COLORS) -> torch.Tensor:
    eye    = np.eye(num_classes, dtype=np.float32)
    onehot = eye[grid]
    return torch.from_numpy(onehot.transpose(2,0,1))

def pad_grids(inp0, out0, tasks, sols, task_id):
    inp0 = np.array(tasks[task_id]['train'][0]['input'])
    out0 = np.array(sols[task_id][0])

    # compute canvas dims
    H, W   = inp0.shape
    Oh, Ow = out0.shape
    # Ch     = max(H, Oh)
    # Cw     = max(W, Ow)

    # pad input
    pad_top_in    = HALF_PATCH
    pad_bottom_in = HALF_PATCH + (GRID_DIM - H)
    pad_left_in   = HALF_PATCH
    pad_right_in  = HALF_PATCH + (GRID_DIM - W)
    padded_in  = np.pad(
        inp0,
        ((pad_top_in, pad_bottom_in), (pad_left_in, pad_right_in)),
        constant_values=PAD_VAL
    )

    # pad output (use Ch, Ow)
    pad_top_out    = HALF_PATCH
    pad_bottom_out = HALF_PATCH + (GRID_DIM - Oh)
    pad_left_out   = HALF_PATCH
    pad_right_out  = HALF_PATCH + (GRID_DIM - Ow)
    padded_out = np.pad(
        out0,
        ((pad_top_out, pad_bottom_out), (pad_left_out, pad_right_out)),
        constant_values=PAD_VAL
    )

    assert padded_in.shape == padded_out.shape, "Padding mismatch!"
    print(f"\n[DEBUG] padded shape: {padded_in.shape}")

    return Oh, Ow, padded_in, padded_out

def create_tensors(padded_in, padded_out):
    # tensors
    x_full = to_onehot(padded_in).unsqueeze(0).to(DEVICE)        # (1,C,Hp,Wp)
    y_full = torch.from_numpy(padded_out).long().unsqueeze(0).to(DEVICE)  # (1,Hp,Wp)
    _, C, Hp, Wp = x_full.shape

    return x_full, y_full, C, Hp, Wp

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
    def __init__(self, num_colors, Hp, Wp, embed_dim=64, nhead=4, num_layers=2):
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

def build_models(C, Hp, Wp):
    cnn = PatchCNN(C).to(DEVICE)
    transformer  = GridTransformer(NUM_COLORS, Hp, Wp).to(DEVICE)
    fusion = FusionModel(cnn, transformer).to(DEVICE)

    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_VAL)

    opt_p = torch.optim.Adam(cnn.parameters(), lr=LR)
    opt_g = torch.optim.Adam(transformer.parameters(),  lr=LR)
    opt_f = torch.optim.Adam(fusion.parameters(), lr=LR)
    return cnn, transformer, fusion, loss_fn, opt_p, opt_g, opt_f

# ── 4) Train PatchCNN ─────────────────────────────────────────────────────────
def train_cnn(model, optimizer, loss_fn, x_full, y_full, C):
    print("\nTraining PatchCNN")
    losses = []
    for epoch in range(1, EPOCHS+1):
        model.train()
        optimizer.zero_grad()
        logits = model(x_full)                 # (1,C,Hp,Wp)
        flat_logits = logits.permute(0,2,3,1).reshape(-1,C)
        flat_labels = y_full.reshape(-1)
        loss = loss_fn(flat_logits, flat_labels)
        losses.append(loss.item())
        # print(f"[PatchCNN] epoch {epoch}, loss={loss.item():.4f}")
        loss.backward()
        optimizer.step()
    avg_loss = sum(losses) / len(losses)
    print(f"Avg Loss: {avg_loss:.4f}")
    torch.save(model.state_dict(), "cnn.pth")
    return avg_loss

# ── 5) Train GridTransformer ──────────────────────────────────────────────────
def train_transformer(model, optimizer, loss_fn, x_full, y_full, C):
    print(f"\nTraining GridTransformer")
    losses = []
    for epoch in range(1, EPOCHS+1):
        model.train()
        optimizer.zero_grad()
        logits = model(x_full)
        flat_logits = logits.permute(0,2,3,1).reshape(-1,C)
        flat_labels = y_full.reshape(-1)
        loss = loss_fn(flat_logits, flat_labels)
        losses.append(loss.item())
        # print(f"[GridTrans] epoch {epoch}, loss={loss.item():.4f}")
        loss.backward()
        optimizer.step()
    avg_loss = sum(losses) / len(losses)
    print(f"Avg Loss: {avg_loss:.4f}")
    torch.save(model.state_dict(), "transformer.pth")
    return avg_loss

# ── 6) Train FusionModel ──────────────────────────────────────────────────────
def train_fusion(model, optimizer, loss_fn, x_full, y_full, C):
    print("\nTraining FusionModel")
    losses = []
    for epoch in range(1, EPOCHS+1):
        model.train()
        optimizer.zero_grad()
        logits = model(x_full)
        flat_logits = logits.permute(0,2,3,1).reshape(-1,C)
        flat_labels = y_full.reshape(-1)
        loss = loss_fn(flat_logits, flat_labels)
        losses.append(loss.item())
        # print(f"[Fusion] epoch {epoch}, loss={loss.item():.4f}")
        loss.backward()
        optimizer.step()
    avg_loss = sum(losses) / len(losses)
    print(f"Avg Loss: {avg_loss:.4f}")
    torch.save(model.state_dict(), "fusion.pth")
    return avg_loss

# ── 7) Final debug & accuracy ─────────────────────────────────────────────────
def evaluate(model, x_full, Oh, Ow, out0):
    model.eval()
    with torch.no_grad():
        pred = model(x_full).argmax(1)[0].cpu().numpy()
        cropped_pred = pred[HALF_PATCH: Oh + HALF_PATCH, HALF_PATCH: Ow + HALF_PATCH]

    print(f"\n[DEBUG] Predicted grid:\n{cropped_pred}")
    print(f"[DEBUG] Ground truth:\n{out0}")

    # exact match?
    grid_match     = np.array_equal(cropped_pred, out0)
    pixel_correct  = (cropped_pred == out0).sum()
    pixel_total    = out0.size
    pixel_accuracy = pixel_correct / pixel_total

    print(f"\nGrid exact‑match: {grid_match}")
    print(f"Pixel accuracy: {pixel_correct}/{pixel_total} = {pixel_accuracy:.2%}")
    
    return int(grid_match), pixel_accuracy

def main(train_tasks, train_sols, eval_tasks, eval_sols, quick_debug=False):
    C = NUM_COLORS
    Hp = GRID_DIM + PATCH_SIZE - 1
    Wp = GRID_DIM + PATCH_SIZE - 1

    cnn, transformer, fusion, loss_fn, opt_p, opt_g, opt_f = build_models(C, Hp, Wp)

    task_count = 0
    grid_match_history = []
    pixel_accuracy_history = []

    cnn_losses = []
    transformer_losses = []
    fusion_losses = []

    for task_id, task in train_tasks.items():
        print(f"[TASK]: {task_id}, [COUNT]: {task_count}")
        for i, pair in enumerate(task['train']):
            inp = np.array(np.array(pair['input']))
            out = np.array(np.array(pair['output']))

            Oh, Ow, padded_in, padded_out = pad_grids(inp, out, train_tasks, train_sols, task_id)
            x_full, y_full, _, _, _ = create_tensors(padded_in, padded_out)

            cnn_loss = train_cnn(cnn, opt_p, loss_fn, x_full, y_full, C)
            transformer_loss = train_transformer(transformer, opt_g, loss_fn, x_full, y_full, C)
            fusion_loss = train_fusion(fusion, opt_f, loss_fn, x_full, y_full, C)

            cnn_losses.append(cnn_loss)
            transformer_losses.append(transformer_loss)
            fusion_losses.append(fusion_loss)

        if task_count % 5 == 0: # only eval every 5 IDs 
            for i, pair in enumerate(task['test']):
                inp = np.array(pair['input'])
                out = np.array(train_sols[task_id][0])  # test labels follow train labels in sols

                Oh, Ow, padded_in, padded_out = pad_grids(inp, out, train_tasks, train_sols, task_id)
                x_val, _, _, _, _ = create_tensors(padded_in, padded_out)

                grid_match, pixel_accuracy = evaluate(fusion, x_val, Oh, Ow, out)

                grid_match_history.append(grid_match)
                pixel_accuracy_history.append(pixel_accuracy)

        task_count += 1

        if task_count >= 50:
            break

        if quick_debug:
            return
    print(f"[GRID MATCHES]: {sum(grid_match_history)}/{len(grid_match_history)}\n")
    print(f"[AVG PIXEL ACCURACY]: {sum(pixel_accuracy_history)/len(pixel_accuracy_history)}\n")
        
    # evaluate
    eval_grid_match_history = []
    eval_pixel_accuracy_history = []
    for task_id, task in eval_tasks.items():
        for i, pair in enumerate(task['test']):
            inp = np.array(pair['input'])
            out = np.array(eval_sols[task_id][0])

            Oh, Ow, padded_in, padded_out = pad_grids(inp, out, eval_tasks, eval_sols, task_id)
            x_full, y_full, _, _, _ = create_tensors(padded_in, padded_out)

            eval_grid_match, eval_pixel_accuracy = evaluate(fusion, x_full, Oh, Ow, out)

            eval_grid_match_history.append(eval_grid_match)
            eval_pixel_accuracy_history.append(eval_pixel_accuracy)
    print(f"[GRID MATCHES]: {sum(eval_grid_match_history)}/{len(eval_grid_match_history)}\n")
    print(f"[AVG PIXEL ACCURACY]: {sum(eval_pixel_accuracy_history)/len(eval_pixel_accuracy_history)}\n")

if __name__ == '__main__':
    with open("data/arc-agi_training_challenges.json") as f:
        train_tasks = json.load(f)
    with open("data/arc-agi_training_solutions.json") as f:
        train_sols  = json.load(f)
    with open("data/arc-agi_evaluation_challenges.json") as f:
        eval_tasks = json.load(f)
    with open("data/arc-agi_evaluation_solutions.json") as f:
        eval_sols  = json.load(f)
    
    main(train_tasks, train_sols, eval_tasks, eval_sols, quick_debug=False)
