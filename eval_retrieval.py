"""Step 4a: retrieval hit-rate@k. Did the right drug + section appear in the top k chunks?"""
import argparse
import json
import os

import rag


def hit_rate(data, k, auto_filter, db_dir):
    hits = 0
    for ex in data:
        docs = rag.retrieve(ex["question"], k=k, persist_dir=db_dir, auto_filter=auto_filter)
        hits += any(d.metadata["drug"] == ex["drug"] and d.metadata["section"] == ex["section"]
                    for d in docs)
    return hits / len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="chroma_db")
    ap.add_argument("--eval", default="eval_set.json")
    args = ap.parse_args()

    with open(args.eval, encoding="utf-8") as f:
        data = json.load(f)

    lines = [f"Index `{args.db}` | {len(data)} questions", "",
             "| k | No drug filter | Auto drug filter |", "|---|---|---|"]
    for k in (2, 4, 6):
        a = hit_rate(data, k, False, args.db)
        b = hit_rate(data, k, True, args.db)
        lines.append(f"| {k} | {a:.2f} | {b:.2f} |")

    out = "\n".join(lines)
    print(out)
    os.makedirs("results", exist_ok=True)
    with open(f"results/retrieval_{os.path.basename(args.db)}.md", "w", encoding="utf-8") as f:
        f.write(out + "\n")


if __name__ == "__main__":
    main()
