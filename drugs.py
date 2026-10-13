import json
import os
import re

DRUGS = [
    "ibuprofen", "acetaminophen", "aspirin", "naproxen", "diclofenac", "metformin",
    "atorvastatin", "simvastatin", "amlodipine", "lisinopril", "losartan", "metoprolol",
    "furosemide", "warfarin", "clopidogrel", "omeprazole", "pantoprazole", "amoxicillin",
    "azithromycin", "ciprofloxacin", "doxycycline", "cetirizine", "loratadine",
    "montelukast", "levothyroxine", "sertraline", "fluoxetine", "alprazolam", "diazepam",
    "tramadol", "gabapentin", "prednisone", "sildenafil", "tamsulosin",
]

# Brand / alternate names -> the generic name used in the index.
ALIASES = {
    "paracetamol": "acetaminophen", "tylenol": "acetaminophen", "advil": "ibuprofen",
    "motrin": "ibuprofen", "aleve": "naproxen", "coumadin": "warfarin",
    "lipitor": "atorvastatin", "zocor": "simvastatin", "viagra": "sildenafil",
    "prozac": "fluoxetine", "zoloft": "sertraline", "xanax": "alprazolam",
    "valium": "diazepam", "neurontin": "gabapentin", "prilosec": "omeprazole",
    "plavix": "clopidogrel", "zithromax": "azithromycin", "glucophage": "metformin",
    "norvasc": "amlodipine", "synthroid": "levothyroxine", "singulair": "montelukast",
    "zyrtec": "cetirizine", "claritin": "loratadine", "lasix": "furosemide",
}


def detect_drugs(text):
    """Return the generic drug names mentioned in a question, in order of appearance."""
    t = text.lower()
    hits = []
    for name in list(DRUGS) + list(ALIASES):
        m = re.search(rf"\b{re.escape(name)}\b", t)
        if m:
            hits.append((m.start(), ALIASES.get(name, name)))
    hits.sort()
    found = []
    for _, d in hits:
        if d not in found:
            found.append(d)
    return found


def available_drugs(labels_path="data/labels.jsonl"):
    """Drugs that actually made it into the data file (falls back to the full list)."""
    if not os.path.exists(labels_path):
        return sorted(DRUGS)
    seen = set()
    with open(labels_path, encoding="utf-8") as f:
        for line in f:
            seen.add(json.loads(line)["drug"])
    return sorted(seen) or sorted(DRUGS)
