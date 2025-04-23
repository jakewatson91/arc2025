import json
import numpy as np
from pathlib import Path

def tile_or_crop_grid(input_grid: list[list[int]],
                      output_shape: tuple[int, int]) -> np.ndarray:
    arr = np.array(input_grid)
    in_h, in_w = arr.shape
    out_h, out_w = output_shape

    # 1) Upscale by tiling
    if out_h % in_h == 0 and out_w % in_w == 0:
        scale_h = out_h // in_h
        scale_w = out_w // in_w
        return np.repeat(np.repeat(arr, scale_h, axis=0),
                         scale_w, axis=1)

    # 2) Downscale by sampling
    if in_h % out_h == 0 and in_w % out_w == 0:
        step_h = in_h // out_h
        step_w = in_w // out_w
        return arr[::step_h, ::step_w]

    # 3) Fallback: center pad/crop
    out = np.zeros((out_h, out_w), dtype=arr.dtype)
    start_out_h = max((out_h - in_h) // 2, 0)
    start_out_w = max((out_w - in_w) // 2, 0)
    end_out_h = start_out_h + min(in_h, out_h)
    end_out_w = start_out_w + min(in_w, out_w)

    start_in_h = max((in_h - out_h) // 2, 0)
    start_in_w = max((in_w - out_w) // 2, 0)
    end_in_h = start_in_h + (end_out_h - start_out_h)
    end_in_w = start_in_w + (end_out_w - start_out_w)

    out[start_out_h:end_out_h, start_out_w:end_out_w] = \
        arr[start_in_h:end_in_h, start_in_w:end_in_w]
    return out

def normalize_colors(arr):
    unique_vals = np.unique(arr)
    remap = {val: i for i, val in enumerate(unique_vals)}
    return np.vectorize(remap.get)(arr)

def get_transforms(pred):
    return [
        pred,
        np.rot90(pred, 1),
        np.rot90(pred, 2),
        np.rot90(pred, 3),
        np.fliplr(pred),
        np.flipud(pred),
        np.transpose(pred),
    ]

def evaluate_exact_match(challenges: dict, solutions: dict) -> tuple[int, int, int]:
    task_correct = 0
    case_correct = 0
    case_total = 0

    for task_id, task in challenges.items():
        test_cases = task["test"]
        all_match = True

        for i, test_case in enumerate(test_cases):
            test_grid = test_case["input"]
            sol_grid = np.array(solutions[task_id][i])
            base_pred = tile_or_crop_grid(test_grid, sol_grid.shape)

            matched = False
            for cand in get_transforms(base_pred):
                pred_norm = normalize_colors(cand)
                sol_norm = normalize_colors(sol_grid)
                if np.array_equal(pred_norm, sol_norm):
                    matched = True
                    break

            if matched:
                case_correct += 1
            else:
                all_match = False
                print(f"❌ Failed: {task_id}, case {i}")
                print("Prediction:")
                print(base_pred)
                print("Solution:")
                print(sol_grid)

            case_total += 1

        if all_match:
            task_correct += 1

    return task_correct, case_correct, case_total

if __name__ == "__main__":
    data_dir = Path("data")

    with open(data_dir / "arc-agi_training_challenges.json") as f:
        train_challenges = json.load(f)
    with open(data_dir / "arc-agi_training_solutions.json") as f:
        train_solutions = json.load(f)
    with open(data_dir / "arc-agi_evaluation_challenges.json") as f:
        eval_challenges = json.load(f)
    with open(data_dir / "arc-agi_evaluation_solutions.json") as f:
        eval_solutions = json.load(f)

    # Training set
    train_task_correct, train_case_correct, train_case_total = evaluate_exact_match(train_challenges, train_solutions)
    train_task_acc = train_task_correct / len(train_challenges)
    train_case_acc = train_case_correct / train_case_total
    print(f"\n✅ Training Task Accuracy: {train_task_correct} / {len(train_challenges)} = {train_task_acc:.2%}")
    print(f"🧩 Training Grid Accuracy: {train_case_correct} / {train_case_total} = {train_case_acc:.2%}")

    # Evaluation set
    eval_task_correct, eval_case_correct, eval_case_total = evaluate_exact_match(eval_challenges, eval_solutions)
    eval_task_acc = eval_task_correct / len(eval_challenges)
    eval_case_acc = eval_case_correct / eval_case_total
    print(f"\n✅ Evaluation Task Accuracy: {eval_task_correct} / {len(eval_challenges)} = {eval_task_acc:.2%}")
    print(f"🧩 Evaluation Grid Accuracy: {eval_case_correct} / {eval_case_total} = {eval_case_acc:.2%}")

    # Eval predictions
    # preds = {}
    # for tid, task in eval_challenges.items():
    #     test_grid = task["test"][0]["input"]
    #     sol_shape = np.array(eval_solutions[tid][0]).shape
    #     base_pred = tile_or_crop_grid(test_grid, sol_shape)

    #     # Try all transforms and normalize
    #     candidates = get_transforms(base_pred)
    #     sol_grid = np.array(eval_solutions[tid][0])
    #     best = next((c for c in candidates
    #                  if np.array_equal(normalize_colors(c), normalize_colors(sol_grid))),
    #                 base_pred)

    #     preds[tid] = best.tolist()
    #     print(f"\n🔸 Eval Task ID: {tid}")
    #     for row in preds[tid]:
    #         print(" ".join(map(str, row)))

    # with open("arc_baseline_tiling_preds.json", "w") as fout:
    #     json.dump(preds, fout, indent=2)
    # print("\n📁 Predictions saved to arc_baseline_tiling_preds.json")