# import json

# #simply pull the data
# training_path = "data/arc-agi_training_challenges.json"
# with open(training_path, 'r') as file:
#     training_json = json.load(file)
# #Only Pull One
# print(training_json["00576224"])

import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# 1) Load one ARC example
with open("data/arc-agi_training_challenges.json") as f:
    train_tasks = json.load(f)
with open("data/arc-agi_training_solutions.json") as f:
    train_sols = json.load(f)

# pick one task
task_id = list(train_tasks.keys())[0]
task = train_tasks[task_id]
test = train_tasks[task_id]["test"]
truth = train_sols[task_id][0]
# truth = train_sols[task_id]["test"]["output"]

# helper to serialize/deserialize
def grid_to_str(g):
    return ";".join(",".join(str(c) for c in row) for row in g)

def str_to_grid(s):
    return [list(map(int,row.split(","))) 
            for row in s.strip().split(";")]

# 2) build few-shot prompt
# prompt = []
# for ex in task["train"]:
#     prompt.append("Input:\n" + grid_to_str(ex["input"]))
#     prompt.append("Output:\n" + grid_to_str(ex["output"]))
# prompt.append("Input:\n" + grid_to_str(task["test"][0]["input"]))
# prompt.append("Output:\nWrite a function that turns the input grid into the output grid```python\ndef solve(input_grid):\n    # TODO: your implementation here\n```")

# prompt = "\n\n".join(prompt)

# 3) load a small model
MODEL = "Salesforce/codet5-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
gen       = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)

def make_code_prompt(train, test_input):
    lines = [
        "# You are a Python coding assistant.  ",
        "# Write a function solve(input_grid) that transforms the input into the correct output.",
        "# Use only valid Python and NumPy.",
        "```python",
    ]
    # few‑shot examples
    for ex in train:
        lines.append(f"# Example:")
        lines.append(f"inp = {ex['input']}")
        lines.append(f"out = {ex['output']}")
    # now the signature
    lines += [
        "",
        "def solve(input_grid):",
        "    # your code here",
        "    pass",
        "",
        "# test it on:",
        f"test_input = {test_input}",
        "",
        "print(solve(test_input))",
        "```",
    ]
    return "\n".join(lines)

# build prompt
prompt = make_code_prompt(
    train=[{"input": [[0,1],[1,0]], "output": [[1,0],[0,1]]},   # replace with your real few shots
           # …
          ],
    test_input=[[1,1],[0,0]]
)

input_grid = task["test"][0]["input"]
test_input_grid = [[1,1],[0,0]]
test_output_grid = [[0,0],[1,1]]

# 4) try up to 5 generations
for attempt in range(1,6):
    print(f"\n=== Attempt {attempt} ===")
    out = gen(prompt, max_new_tokens=256, do_sample=False)[0]["generated_text"]
    print("📝 Generated code:\n", out, "\n")

    # 5) try to exec & run
    ns = {}
    try:
        exec(out, ns)  # define solve(...)
        if "solve" not in ns:
            raise RuntimeError("no solve() in generated code")
        # pred = ns["solve"](input_grid)
        pred = ns["solve"](test_input_grid)
    except Exception as e:
        print("⚠️  Exec error:", e)
        feedback = f"# The code failed with {e}\n"
    else:
        print("🚀 Model output:", np.array(pred))
        print("✅ True output: ", np.array(truth))
        # if pred == truth:
        if pred == test_output_grid:
            print("🎉 Success!")
            break
        else:
            feedback = "# The output was incorrect; please fix your solve() implementation.\n"

    # append feedback to prompt for next iteration
    prompt += feedback + "Output:\n"

else:
    print("❌ All attempts failed.")
