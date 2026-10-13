"""Checks that every (drug, section) pair in eval_set.json exists in data/labels.jsonl."""
import json

have = set()
with open("data/labels.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        have.add((r["drug"], r["section"]))

with open("eval_set.json", encoding="utf-8") as f:
    data = json.load(f)

bad = [ex for ex in data if (ex["drug"], ex["section"]) not in have]
for ex in bad:
    print(f"MISSING in data: {ex['drug']} | {ex['section']} -> edit or remove: {ex['question']}")
print(f"\n{len(data) - len(bad)}/{len(data)} eval questions are usable.")
