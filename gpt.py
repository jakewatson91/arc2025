import os
from dotenv import load_dotenv
from openai import OpenAI
# 1) install with: pip install openai

load_dotenv()
client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
# client.api_key = os.getenv("OPENAI_API_KEY")  # or set it directly

# def make_code_prompt(train, test_input):
#     lines = [
#         "You are a Python coding assistant.",
#         "Write a function `solve(input_grid)` that transforms the input into the correct output.",
#         "Use only valid Python and NumPy.",
#         "```python",
#     ]
#     for ex in train:
#         lines += [
#             f"# Example",
#             f"inp = {ex['input']}",
#             f"out = {ex['output']}",
#         ]
#     lines += [
#         "",
#         "def solve(input_grid):",
#         "    # your code here",
#         "    pass",
#         "",
#         "# test it on:",
#         f"test_input = {test_input}",
#         "",
#         "print(solve(test_input))",
#         "```",
#     ]
#     return "\n".join(lines)

# def generate_code_with_gpt35(prompt: str) -> str:
#     resp = client.responses.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {"role": "system", "content": "You write correct, executable Python code."},
#             {"role": "user",   "content": prompt}
#         ],
#         temperature=0.0,
#         max_tokens=300,
#         n=1,
#         stop=["```"]
#     )
#     return resp.choices[0].message.content

# # example few‑shot
# few_shots = [
#     {"input": [[0,1],[1,0]], "output": [[1,0],[0,1]]},
#     {"input": [[1,1],[0,0]], "output": [[0,0],[1,1]]},
# ]

# test_inp = [[1,0,1],[0,1,0]]

# prompt = make_code_prompt(few_shots, test_inp)
# code = generate_code_with_gpt35(prompt)
# print("=== Generated code ===\n", code)

prompt = "multiply each number in the list together: [1, 2, 3]"
def generate_code_with_gpt35(prompt: str) -> str:
    resp = client.responses.create(
        model="gpt-3.5-turbo",
        instructions="""
        You are a Python coding assistant.
        Write a function `solve(task)` that transforms the input into the correct output.
        Use only valid Python and NumPy.
        ```TO-DO: Your python code here
        def solve(task):
            pass```
        """,
        input=f"write an executable python script to solve {prompt}"
    )
    return resp.output[0].content

code = generate_code_with_gpt35(prompt)
print("=== Generated code ===\n", code)

# input_grid = task["test"][0]["input"]
# test_input_grid = [[1,1],[0,0]]
# test_output_grid = [[0,0],[1,1]]

# # 4) try up to 5 generations
# for attempt in range(1,6):
#     print(f"\n=== Attempt {attempt} ===")
#     out = gen(prompt, max_new_tokens=256, do_sample=False)[0]["generated_text"]
#     print("📝 Generated code:\n", out, "\n")

#     # 5) try to exec & run
#     ns = {}
#     try:
#         exec(out, ns)  # define solve(...)
#         if "solve" not in ns:
#             raise RuntimeError("no solve() in generated code")
#         # pred = ns["solve"](input_grid)
#         pred = ns["solve"](test_input_grid)
#     except Exception as e:
#         print("⚠️  Exec error:", e)
#         feedback = f"# The code failed with {e}\n"
#     else:
#         print("🚀 Model output:", np.array(pred))
#         print("✅ True output: ", np.array(truth))
#         # if pred == truth:
#         if pred == test_output_grid:
#             print("🎉 Success!")
#             break
#         else:
#             feedback = "# The output was incorrect; please fix your solve() implementation.\n"

#     # append feedback to prompt for next iteration
#     prompt += feedback + "Output:\n"

# else:
#     print("❌ All attempts failed.")
