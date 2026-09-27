import json

with open("data/candidate_pairs.json", "r") as f:
    candidates = json.load(f)

# Inspect first 5 keys and their value lists
sample_keys = list(candidates.keys())[:5]
for k in sample_keys:
    print(f"Key S1 ID: '{k}' -> Candidates: {candidates[k]}")