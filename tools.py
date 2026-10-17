"""Agent tools: label search (RAG), interaction checker, adverse-event statistics."""
import json
import os
import re
from functools import lru_cache
from typing import Optional

import requests
from langchain_core.tools import tool

import rag
from config import DEFAULT_DB_DIR, LABELS_PATH
from drugs import ALIASES, DRUGS

DB_DIR = DEFAULT_DB_DIR      # tests override these two
LABELS = LABELS_PATH
EVENT_URL = "https://api.fda.gov/drug/event.json"

# Drug classes and pharmacological mechanisms a label may use instead of naming the other drug.
CLASS_TERMS = {
    "ibuprofen": ["nsaid", "nsaids", "nonsteroidal"], "naproxen": ["nsaid", "nsaids", "nonsteroidal"],
    "diclofenac": ["nsaid", "nsaids", "nonsteroidal"], "aspirin": ["nsaid", "nsaids", "salicylate", "salicylates"],
    "warfarin": ["anticoagulant", "anticoagulants", "coumarin"], "clopidogrel": ["antiplatelet", "antiplatelets"],
    "lisinopril": ["ace inhibitor", "ace inhibitors"], "losartan": ["angiotensin receptor blocker", "arb", "arbs"],
    "simvastatin": ["statin", "statins", "cyp3a4", "hmg-coa reductase inhibitor"],
    "atorvastatin": ["statin", "statins", "cyp3a4", "hmg-coa reductase inhibitor"],
    "amlodipine": ["calcium channel blocker", "calcium channel blockers", "ccb", "ccbs", "dihydropyridine"],
    "metoprolol": ["beta blocker", "beta blockers", "beta-blocker", "beta-blockers"],
    "furosemide": ["loop diuretic", "diuretic", "diuretics"],
    "prednisone": ["corticosteroid", "corticosteroids", "steroid", "steroids"],
    "sertraline": ["ssri", "ssris", "serotonergic"], "fluoxetine": ["ssri", "ssris", "serotonergic"],
    "tramadol": ["opioid", "opioids", "serotonergic"],
    "alprazolam": ["benzodiazepine", "benzodiazepines"], "diazepam": ["benzodiazepine", "benzodiazepines"],
    "omeprazole": ["proton pump inhibitor", "ppi", "ppis"], "pantoprazole": ["proton pump inhibitor", "ppi", "ppis"],
    "ciprofloxacin": ["fluoroquinolone", "fluoroquinolones", "quinolone", "quinolones"],
    "doxycycline": ["tetracycline", "tetracyclines"],
    "azithromycin": ["macrolide", "macrolides"],
    "amoxicillin": ["penicillin", "penicillins", "beta-lactam", "beta-lactams"],
}

# Sections scanned for drug interactions, dosage caps, and contraindications
SAFETY_SECTIONS = [
    "Drug Interactions",
    "Dosage and Administration",
    "Warnings and Precautions",
    "Contraindications",
]


def normalize(name):
    """Map a user-supplied drug name or brand to the generic name in the index (None if unknown)."""
    n = (name or "").strip().lower()
    n = ALIASES.get(n, n)
    return n if n in DRUGS else None


@lru_cache(maxsize=4)
def _sections(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[(r["drug"], r["section"])] = r["text"]
    return out


def _mentions(text, other):
    terms = [other] + [k for k, v in ALIASES.items() if v == other] + CLASS_TERMS.get(other, [])
    hits = []
    for t in terms:
        for m in re.finditer(rf"\b{re.escape(t)}\b", text, flags=re.I):
            hits.append((t, m.start()))
    return sorted(hits, key=lambda h: h[1])


def _excerpt(text, pos, width=220):
    return "..." + text[max(0, pos - width): pos + width] + "..."


@tool
def search_labels(query: str, drug: Optional[str] = None) -> str:
    """Semantic search over FDA drug label sections (indications, dosage, contraindications,
    warnings, interactions, adverse reactions). Use for any factual question about a medicine.
    Optionally pass a single drug name to restrict the search to that drug's label."""
    d = normalize(drug) if drug else None
    docs = rag.retrieve(query, drug=d, k=4, persist_dir=DB_DIR)
    if not docs:
        return "No label text found."
    return "\n\n".join(x.page_content for x in docs)


@tool
def check_interaction(drug_a: str, drug_b: str) -> str:
    """Check whether the FDA label of either drug mentions the other (by name, brand or drug class)
    across its Drug Interactions, Dosage, Warnings, or Contraindications sections.
    Use for any question like 'can I take A with B'."""
    a, b = normalize(drug_a), normalize(drug_b)
    unknown = [raw for raw, norm in ((drug_a, a), (drug_b, b)) if norm is None]
    if unknown:
        return f"Not in the index: {', '.join(unknown)}. Indexed drugs: {', '.join(sorted(DRUGS))}."
    if a == b:
        return "Both names refer to the same drug; ask about two different drugs."
    secs = _sections(LABELS)
    lines, found = [], False
    for x, y in ((a, b), (b, a)):
        drug_hits = []
        for sec in SAFETY_SECTIONS:
            text = secs.get((x, sec))
            if text:
                hits = _mentions(text, y)
                if hits:
                    found = True
                    term, pos = hits[0]
                    drug_hits.append(f"[{x} | {sec}] mentions '{term}': {_excerpt(text, pos)}")
        if drug_hits:
            lines.extend(drug_hits)
        else:
            lines.append(f"[{x} | Drug Interactions] does not mention {y} or its drug class.")

    lines.append("Result: " + ("at least one label flags this pair." if found else
                 "could not verify interaction in official package insert sections. This is NOT proof of safety; absence of an explicit warning is NOT proof that the combination is safe."))
    return "\n".join(lines)


@tool
def adverse_event_stats(drug: str) -> str:
    """Most frequently reported adverse reactions for a drug in the FDA FAERS database (via openFDA).
    Use when asked which side effects are most reported in real-world reports."""
    d = normalize(drug)
    if d is None:
        return f"Not in the index: {drug}."
    params = {"search": f'patient.drug.openfda.generic_name:"{d.upper()}"',
              "count": "patient.reaction.reactionmeddrapt.exact", "limit": 5}
    if os.getenv("OPENFDA_API_KEY"):
        params["api_key"] = os.getenv("OPENFDA_API_KEY")
    try:
        r = requests.get(EVENT_URL, params=params, timeout=20)
        r.raise_for_status()
        results = r.json().get("results", [])
    except (requests.RequestException, ValueError) as e:
        return f"openFDA is unavailable right now ({type(e).__name__})."
    if not results:
        return f"No adverse event reports found for {d}."
    rows = ", ".join(f"{x['term']} ({x['count']})" for x in results)
    return (f"[{d} | FAERS reports] Most frequently reported reactions: {rows}. "
            "Reports are voluntary and do not prove the drug caused the reaction.")


import itertools
from drugs import BRAND_NOTES, detect_drugs


@tool
def check_medication_list(medications: str) -> str:
    """Analyze a multi-drug regimen (3 or more medications) for all pairwise drug-drug interactions.
    Input: comma-, semicolon-, or newline-separated list of drug names or brands (e.g. 'warfarin, aspirin, metformin, lisinopril, atorvastatin'
    or Indian brands like 'ecosprin, dolo, combiflam, glycomet').
    Checks all unique pairs across FDA label Drug Interactions, Dosage, Warnings, and Contraindications sections."""
    # Detect drugs via detect_drugs
    found_drugs = detect_drugs(medications)

    # Also check raw split items in case of comma-separated inputs
    raw_tokens = re.split(r"[,;\n\+]+", medications)
    for raw in raw_tokens:
        norm = normalize(raw)
        if norm and norm not in found_drugs:
            found_drugs.append(norm)

    if len(found_drugs) < 2:
        return (f"Need at least 2 indexed medications to check interactions. Found: {', '.join(found_drugs) if found_drugs else 'none'}. "
                f"Indexed drugs: {', '.join(sorted(DRUGS))}.")

    secs = _sections(LABELS)
    flagged = []
    unflagged = []

    for a, b in itertools.combinations(found_drugs, 2):
        pair_flagged = False
        excerpts = []
        for x, y in ((a, b), (b, a)):
            for sec in SAFETY_SECTIONS:
                text = secs.get((x, sec))
                if text:
                    hits = _mentions(text, y)
                    if hits:
                        pair_flagged = True
                        term, pos = hits[0]
                        excerpts.append(f"[{x} | {sec}] mentions '{term}': {_excerpt(text, pos, width=150)}")

        if pair_flagged:
            flagged.append((a, b, " | ".join(excerpts)))
        else:
            unflagged.append((a, b))

    # Identify any brand notes relevant to the query
    q_lower = medications.lower()
    active_brand_notes = [note for brand, note in BRAND_NOTES.items() if brand in q_lower]

    out_lines = [
        f"### Polypharmacy Interaction Audit ({len(found_drugs)} medications, {len(found_drugs)*(len(found_drugs)-1)//2} pairs analyzed)",
        f"**Medications evaluated:** {', '.join(found_drugs).title()}",
        ""
    ]

    if active_brand_notes:
        out_lines.append("**Brand / Formulation Notes:**")
        for bn in active_brand_notes:
            out_lines.append(f"- {bn}")
        out_lines.append("")

    if flagged:
        out_lines.append(f"#### 🚨 Flagged Interactions ({len(flagged)} pair(s))")
        for a, b, exc in flagged:
            out_lines.append(f"- **{a.title()} + {b.title()}**: {exc}")
        out_lines.append("")

    if unflagged:
        out_lines.append(f"#### ℹ️ Could Not Verify in Package Insert ({len(unflagged)} pair(s))")
        for a, b in unflagged:
            out_lines.append(f"- {a.title()} + {b.title()}")
        out_lines.append("\n*Important clinical note:* Could not verify interaction in official package insert sections (Drug Interactions, Dosage, Warnings, Contraindications). Absence of an explicit warning in the FDA package insert is NOT proof of safety. Always review patient renal/hepatic function and consult a physician or pharmacist.")

    return "\n".join(out_lines)


TOOLS = [search_labels, check_interaction, adverse_event_stats, check_medication_list]
