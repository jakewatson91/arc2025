import json
import numpy as np
import torch
import torch.nn as nn

from models import PatchCNN, GridTransformer, FusionModel
from plot_utils import plot_training_losses

# ── Hyperparameters ────────────────────────────────────────────────────────────
PATCH_SIZE    = 11
GRID_DIM      = 30 # max grid size = 900
NUM_COLORS    = 10
LR            = 1e-5
EPOCHS        = 100
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HALF_PATCH    = PATCH_SIZE // 2
PAD_VAL       = -1

# ── One‑hot encoding helper ────────────────────────────────────────────────────
def to_onehot(grid: np.ndarray, num_classes: int = NUM_COLORS) -> torch.Tensor:
    eye    = np.eye(num_classes, dtype=np.float32)
    onehot = eye[grid]
    return torch.from_numpy(onehot.transpose(2,0,1))

# padding breaks grids into smaller squares to focus on local patterns
# this gives the model much more tasks to train on
def pad_grids(original, target):

    # compute canvas dims
    H, W   = original.shape
    target_H, target_W = target.shape

    # pad input
    pad_top_in    = HALF_PATCH
    pad_bottom_in = HALF_PATCH + (GRID_DIM - H)
    pad_left_in   = HALF_PATCH
    pad_right_in  = HALF_PATCH + (GRID_DIM - W)
    padded_in  = np.pad(
        original,
        ((pad_top_in, pad_bottom_in), (pad_left_in, pad_right_in)),
        constant_values=PAD_VAL
    )

    # pad output (use Ch, Ow)
    pad_top_out    = HALF_PATCH
    pad_bottom_out = HALF_PATCH + (GRID_DIM - target_H)
    pad_left_out   = HALF_PATCH
    pad_right_out  = HALF_PATCH + (GRID_DIM - target_W)
    padded_out = np.pad(
        target,
        ((pad_top_out, pad_bottom_out), (pad_left_out, pad_right_out)),
        constant_values=PAD_VAL
    )

    # assert padded_in.shape == padded_out.shape, "Padding mismatch!"
    print(f"\nPadded shape: {padded_in.shape}")

    return target_H, target_W, padded_in, padded_out

def create_tensors(padded_in, padded_out):
    # padded tensors - one-hot encoded input, padded output
    X_full = to_onehot(padded_in).unsqueeze(0).to(DEVICE)        # (1,C,H,W)
    y_full = torch.from_numpy(padded_out).long().unsqueeze(0).to(DEVICE)  # (1,H,W)
    _, C, H, W = X_full.shape

    return X_full, y_full, C, H, W

def build_models(C, H, W):
    cnn = PatchCNN(C).to(DEVICE)
    transformer  = GridTransformer(NUM_COLORS, H, W).to(DEVICE)
    fusion = FusionModel(cnn, transformer).to(DEVICE)

    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_VAL)

    opt_patch = torch.optim.Adam(cnn.parameters(), lr=LR)
    opt_transformer = torch.optim.Adam(transformer.parameters(),  lr=LR)
    opt_fusion = torch.optim.Adam(fusion.parameters(), lr=LR)
    return cnn, transformer, fusion, loss_fn, opt_patch, opt_transformer, opt_fusion

# ── 4) Train PatchCNN ─────────────────────────────────────────────────────────
def train_cnn(model, optimizer, loss_fn, X, y, C, save=True):
    print("\nTraining PatchCNN")
    losses = []
    for epoch in range(1, EPOCHS+1):
        model.train()
        optimizer.zero_grad()
        logits = model(X)                 # (1,C,Hp,Wp)
        flat_logits = logits.permute(0,2,3,1).reshape(-1,C)
        flat_labels = y.reshape(-1)
        loss = loss_fn(flat_logits, flat_labels)
        losses.append(loss.item())
        # print(f"[PatchCNN] epoch {epoch}, loss={loss.item():.4f}")
        loss.backward()
        optimizer.step()
    avg_loss = sum(losses) / len(losses)
    print(f"Avg Loss: {avg_loss:.4f}")
    if save:
        torch.save(model.state_dict(), "cnn2.pth")
    return avg_loss

# ── 5) Train GridTransformer ──────────────────────────────────────────────────
def train_transformer(model, optimizer, loss_fn, X, y, C, save=True):
    print(f"\nTraining GridTransformer")
    losses = []
    for epoch in range(1, EPOCHS+1):
        model.train()
        optimizer.zero_grad()
        logits = model(X)
        flat_logits = logits.permute(0,2,3,1).reshape(-1,C)
        flat_labels = y.reshape(-1)
        loss = loss_fn(flat_logits, flat_labels)
        losses.append(loss.item())
        # print(f"[GridTrans] epoch {epoch}, loss={loss.item():.4f}")
        loss.backward()
        optimizer.step()
    avg_loss = sum(losses) / len(losses)
    print(f"Avg Loss: {avg_loss:.4f}")
    if save:
        torch.save(model.state_dict(), "transformer2.pth")
    return avg_loss

# ── 6) Train FusionModel ──────────────────────────────────────────────────────
def train_fusion(model, optimizer, loss_fn, X, y, C, save=True):
    print("\nTraining FusionModel")
    losses = []
    for epoch in range(1, EPOCHS+1):
        model.train()
        optimizer.zero_grad()
        logits = model(X)
        flat_logits = logits.permute(0,2,3,1).reshape(-1,C)
        flat_labels = y.reshape(-1)
        loss = loss_fn(flat_logits, flat_labels)
        losses.append(loss.item())
        # print(f"[Fusion] epoch {epoch}, loss={loss.item():.4f}")
        loss.backward()
        optimizer.step()
    avg_loss = sum(losses) / len(losses)
    print(f"Avg Loss: {avg_loss:.4f}")
    if save:
        torch.save(model.state_dict(), "fusion2.pth")
    return avg_loss

# ── 7) Final debug & accuracy ─────────────────────────────────────────────────
def evaluate(model, X_full, target_H, target_W, target, original):
    model.eval()
    with torch.no_grad():
        pred = model(X_full).argmax(1)[0].cpu().numpy()
        cropped_pred = pred[HALF_PATCH: target_H + HALF_PATCH, HALF_PATCH: target_W + HALF_PATCH]

    print(f"\nOriginal grid:\n{original}")
    print(f"\nPredicted grid:\n{cropped_pred}")
    print(f"Ground truth:\n{target}")

    # exact match?
    grid_match     = np.array_equal(cropped_pred, target)
    pixel_correct  = (cropped_pred == target).sum()
    pixel_total    = target.size
    pixel_accuracy = pixel_correct / pixel_total

    print(f"\nGrid exact‑match: {grid_match}")
    print(f"Pixel accuracy: {pixel_correct}/{pixel_total} = {pixel_accuracy:.2%}")
    
    return int(grid_match), pixel_accuracy

def train_loop(train_tasks, train_sols, cnn, transformer, fusion, loss_fn, opt_patch, opt_transformer, opt_fusion, C, quick_debug):
    task_count = 0
    grid_match_history = []
    pixel_accuracy_history = []
    cnn_losses = []
    transformer_losses = []
    fusion_losses = []

    for task_id, task in train_tasks.items():
        print(f"[TASK]: {task_id}, [COUNT]: {task_count}")
        for i, pair in enumerate(task['train']):
            original = np.array(np.array(pair['input']))
            target = np.array(np.array(pair['output']))

            _, _, padded_in, padded_out = pad_grids(original, target)
            X, y, _, _, _ = create_tensors(padded_in, padded_out)

            cnn_loss = train_cnn(cnn, opt_patch, loss_fn, X, y, C)
            transformer_loss = train_transformer(transformer, opt_transformer, loss_fn, X, y, C)
            fusion_loss = train_fusion(fusion, opt_fusion, loss_fn, X, y, C)

            cnn_losses.append(cnn_loss)
            transformer_losses.append(transformer_loss)
            fusion_losses.append(fusion_loss)

        if task_count % 5 == 0:
            for i, pair in enumerate(task['test']):
                original = np.array(pair['input'])
                target = np.array(train_sols[task_id][0])

                target_H, target_W, padded_in, padded_out = pad_grids(original, target)
                X, _, _, _, _ = create_tensors(padded_in, padded_out)

                grid_match, pixel_accuracy = evaluate(fusion, X, target_H, target_W, target, original)

                grid_match_history.append(grid_match)
                pixel_accuracy_history.append(pixel_accuracy)

        task_count += 1

        if task_count >= 5:
            break

        if quick_debug:
            break

    plot_training_losses(cnn_losses, transformer_losses, fusion_losses)
    return grid_match_history, pixel_accuracy_history

def eval_loop(eval_tasks, eval_sols, cnn, transformer, fusion, loss_fn, opt_patch, opt_transformer, opt_fusion, C):
    eval_grid_match_history = []
    eval_pixel_accuracy_history = []

    cnn.load_state_dict(torch.load("cnn2.pth", map_location=DEVICE))
    transformer.load_state_dict(torch.load("transformer2.pth", map_location=DEVICE))
    fusion.load_state_dict(torch.load("fusion2.pth", map_location=DEVICE))

    for task_id, task in eval_tasks.items():
        for i, pair in enumerate(task['train']):
            original = np.array(np.array(pair['input']))
            target = np.array(np.array(pair['output']))

            _, _, padded_in, padded_out = pad_grids(original, target)
            X, y, _, _, _ = create_tensors(padded_in, padded_out)

            train_cnn(cnn, opt_patch, loss_fn, X, y, C, save=False)
            train_transformer(transformer, opt_transformer, loss_fn, X, y, C, save=False)
            train_fusion(fusion, opt_fusion, loss_fn, X, y, C, save=False)

        for i, pair in enumerate(task['test']):
            original = np.array(pair['input'])
            target = np.array(eval_sols[task_id][0])

            target_H, target_W, padded_in, padded_out = pad_grids(original, target)
            X, _, _, _, _ = create_tensors(padded_in, padded_out)

            eval_grid_match, eval_pixel_accuracy = evaluate(fusion, X, target_H, target_W, target, original)

            eval_grid_match_history.append(eval_grid_match)
            eval_pixel_accuracy_history.append(eval_pixel_accuracy)

    return eval_grid_match_history, eval_pixel_accuracy_history

def main(train_tasks, train_sols, eval_tasks, eval_sols, do_eval=True, quick_debug=False):
    C = NUM_COLORS
    H = GRID_DIM + PATCH_SIZE - 1
    W = GRID_DIM + PATCH_SIZE - 1

    cnn, transformer, fusion, loss_fn, opt_patch, opt_transformer, opt_fusion = build_models(C, H, W)

    grid_match_history, pixel_accuracy_history = train_loop(
        train_tasks, train_sols, cnn, transformer, fusion, loss_fn, opt_patch, opt_transformer, opt_fusion, C, quick_debug
    )

    print(f"[GRID MATCHES]: {sum(grid_match_history)}/{len(grid_match_history)}\n")
    print(f"[AVG PIXEL ACCURACY]: {sum(pixel_accuracy_history)/len(pixel_accuracy_history)}\n")

    if do_eval:
        eval_grid_match_history, eval_pixel_accuracy_history = eval_loop(
            eval_tasks, eval_sols, cnn, transformer, fusion, loss_fn, opt_patch, opt_transformer, opt_fusion, C
        )

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
    
    main(train_tasks, train_sols, eval_tasks, eval_sols, do_eval=True, quick_debug=False)
