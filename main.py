import json

#simply pull the data
training_path = "/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json"
with open(training_path, 'r') as file:
    training_json = json.load(file)
#Only Pull One
print(training_json["00576224"])
