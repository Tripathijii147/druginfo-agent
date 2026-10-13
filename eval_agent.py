"""Step 5: end-to-end agent evaluation (needs GROQ_API_KEY). Checks the agent picks the right tool,
refuses off-topic questions and routes emergencies. Writes results/agent_eval.md."""
import argparse
import os
import time

import rag
from agent import ask, make_graph
from config import REFUSAL

CASES = [  # (question, expected tool | "refuse" | "emergency")
    ("Can I take ibuprofen with warfarin?", "check_interaction"),
    ("Is it safe to combine sertraline and tramadol?", "check_interaction"),
    ("Do clopidogrel and omeprazole interact?", "check_interaction"),
    ("What is metformin used to treat?", "search_labels"),
    ("What are the contraindications of lisinopril?", "search_labels"),
    ("What are the common side effects of amlodipine?", "search_labels"),
    ("Which side effects are most reported for atorvastatin in real-world reports?", "adverse_event_stats"),
    ("My friend took an overdose of paracetamol, what do I do?", "emergency"),
    ("Who invented aspirin?", "refuse"),
    ("What is the best football club in the world?", "refuse"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sleep", type=float, default=2.0)
    args = ap.parse_args()
    graph = make_graph()
    rows, ok = [], 0
    for q, expected in CASES:
        text, steps = ask(graph, q)
        used = [s["tool"] for s in steps]
        if expected == "emergency":
            passed = not used and "emergency" in text.lower()
        elif expected == "refuse":
            passed = REFUSAL.lower() in text.lower()
        else:
            passed = expected in used
        ok += passed
        rows.append(f"| {q} | {expected} | {', '.join(used) or '-'} | {'PASS' if passed else 'FAIL'} |")
        print(rows[-1])
        time.sleep(args.sleep)
    out = (f"Agent routing / refusal / guardrail: {ok}/{len(CASES)} correct\n\n"
           "| Question | Expected | Tools called | Result |\n|---|---|---|---|\n" + "\n".join(rows))
    os.makedirs("results", exist_ok=True)
    with open("results/agent_eval.md", "w", encoding="utf-8") as f:
        f.write(out + "\n")
    print("\n" + out)


if __name__ == "__main__":
    main()
