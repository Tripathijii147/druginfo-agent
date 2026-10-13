"""Step 4b: answer quality. Faithfulness (LLM judge), source hit, and refusal on unanswerable questions."""
import argparse
import csv
import json
import os
import time

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import rag
from config import REFUSAL

UNANSWERABLE = [
    "What is the price of metformin in India?",
    "Which pharmacy sells atorvastatin cheapest?",
    "Who invented aspirin?",
    "What is the best football club in the world?",
]

JUDGE = ChatPromptTemplate.from_messages([
    ("system", "You are a strict grader. Decide whether every factual claim in the ANSWER is "
               "supported by the EXCERPTS. Reply with exactly one word: YES or NO."),
    ("human", "EXCERPTS:\n{context}\n\nANSWER:\n{answer}"),
])


def with_retry(fn, tries=5):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # rate limits, timeouts
            wait = 5 * 2 ** i
            print(f"  {type(e).__name__}; retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("Too many failures; check your API key / model name / rate limits.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="chroma_db")
    ap.add_argument("--eval", default="eval_set.json")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="only run the first N questions")
    ap.add_argument("--sleep", type=float, default=2.0, help="seconds between API calls")
    args = ap.parse_args()

    with open(args.eval, encoding="utf-8") as f:
        data = json.load(f)
    if args.limit:
        data = data[: args.limit]

    judge = JUDGE | rag.get_llm() | StrOutputParser()
    rows, faithful, judged, source_hits, refused_answerable = 0, 0, 0, 0, 0
    os.makedirs("results", exist_ok=True)
    with open("results/faithfulness.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["question", "expected_drug", "expected_section", "source_hit", "refused", "faithful", "answer"])
        for ex in data:
            text, docs = with_retry(lambda: rag.answer(ex["question"], k=args.k, persist_dir=args.db))
            hit = any(d.metadata["drug"] == ex["drug"] and d.metadata["section"] == ex["section"] for d in docs)
            refused = REFUSAL.lower() in text.lower()
            verdict = ""
            if refused:
                refused_answerable += 1
            else:
                context = "\n\n".join(d.page_content for d in docs)
                verdict = with_retry(lambda: judge.invoke({"context": context, "answer": text}))
                judged += 1
                faithful += verdict.strip().upper().startswith("YES")
            rows += 1
            source_hits += hit
            w.writerow([ex["question"], ex["drug"], ex["section"], hit, refused, verdict.strip(), text])
            print(f"[{rows}/{len(data)}] hit={hit} refused={refused} judge={verdict.strip() or '-'}")
            time.sleep(args.sleep)

    print("\nTesting unanswerable questions...")
    refusals = 0
    for q in UNANSWERABLE:
        text, _ = with_retry(lambda: rag.answer(q, k=args.k, persist_dir=args.db))
        refusals += REFUSAL.lower() in text.lower()
        print(f"  refused={REFUSAL.lower() in text.lower()} | {q}")
        time.sleep(args.sleep)

    lines = [
        f"Questions: {rows} | k={args.k}",
        f"- Correct source in retrieved chunks: {source_hits / rows:.0%}",
        f"- Faithfulness (LLM judge, non-refused answers): {faithful / max(judged, 1):.0%} ({faithful}/{judged})",
        f"- Answerable questions wrongly refused: {refused_answerable}/{rows}",
        f"- Unanswerable questions correctly refused: {refusals}/{len(UNANSWERABLE)}",
    ]
    out = "\n".join(lines)
    print("\n" + out)
    with open("results/faithfulness_summary.md", "w", encoding="utf-8") as f:
        f.write(out + "\n")


if __name__ == "__main__":
    main()
