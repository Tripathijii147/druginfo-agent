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
# Includes US and prominent Indian brand names mapped to the 34 indexed FDA generics.
ALIASES = {
    # Paracetamol / Acetaminophen
    "paracetamol": "acetaminophen", "tylenol": "acetaminophen", "dolo": "acetaminophen",
    "dolo 650": "acetaminophen", "dolo-650": "acetaminophen", "crocin": "acetaminophen",
    "calpol": "acetaminophen", "pacimol": "acetaminophen", "fepex": "acetaminophen",

    # NSAIDs
    "advil": "ibuprofen", "motrin": "ibuprofen", "brufen": "ibuprofen", "ibugesic": "ibuprofen",
    "combiflam": "ibuprofen",  # Fixed-dose combination: Ibuprofen 400mg + Paracetamol 325mg
    "aleve": "naproxen",
    "voveran": "diclofenac", "volini": "diclofenac",
    "disprin": "aspirin", "ecosprin": "aspirin", "asprin": "aspirin",

    # Anticoagulants & Antiplatelets
    "coumadin": "warfarin", "warf": "warfarin", "acitrom": "warfarin",
    "plavix": "clopidogrel", "clopilet": "clopidogrel", "deplatt": "clopidogrel",

    # Statins
    "lipitor": "atorvastatin", "atorva": "atorvastatin", "storvas": "atorvastatin", "tonact": "atorvastatin",
    "zocor": "simvastatin", "simvotin": "simvastatin",

    # Antihypertensives
    "norvasc": "amlodipine", "amlong": "amlodipine", "stamlo": "amlodipine",
    "zestril": "lisinopril", "prinivil": "lisinopril",
    "cozaar": "losartan", "losar": "losartan", "telma": "losartan",  # ARB class mapping
    "lopressor": "metoprolol", "met-xl": "metoprolol", "betaloc": "metoprolol",
    "lasix": "furosemide",

    # Antidiabetic
    "glucophage": "metformin", "glycomet": "metformin", "gluconorm": "metformin", "obimet": "metformin",

    # Acid Suppressants (PPIs)
    "prilosec": "omeprazole", "omez": "omeprazole", "omee": "omeprazole",
    "protonix": "pantoprazole", "pantocid": "pantoprazole", "pan 40": "pantoprazole",
    "pan-40": "pantoprazole", "pantosec": "pantoprazole", "pan": "pantoprazole",

    # Antibiotics
    "amoxil": "amoxicillin", "augmentin": "amoxicillin", "moxikind": "amoxicillin", "novamox": "amoxicillin",
    "zithromax": "azithromycin", "azithral": "azithromycin", "zady": "azithromycin",
    "cipro": "ciprofloxacin", "ciplox": "ciprofloxacin", "cifran": "ciprofloxacin",
    "vibramycin": "doxycycline", "doxylab": "doxycycline",

    # Antihistamines & Respiratory
    "zyrtec": "cetirizine", "cetzine": "cetirizine", "alerid": "cetirizine", "okacet": "cetirizine",
    "claritin": "loratadine", "claridin": "loratadine", "lorfast": "loratadine",
    "singulair": "montelukast", "montair": "montelukast", "montair-lc": "montelukast",

    # Thyroid & Endocrine
    "synthroid": "levothyroxine", "thyronorm": "levothyroxine", "eltroxin": "levothyroxine",

    # Neuropsychiatric & Analgesic
    "prozac": "fluoxetine", "flunil": "fluoxetine",
    "zoloft": "sertraline", "daxid": "sertraline",
    "xanax": "alprazolam", "restyl": "alprazolam", "alprax": "alprazolam",
    "valium": "diazepam", "calmpose": "diazepam",
    "ultram": "tramadol", "tramazac": "tramadol",
    "neurontin": "gabapentin", "gabapin": "gabapentin",
    "deltasone": "prednisone", "omnacortil": "prednisone", "wysolone": "prednisone",
    "viagra": "sildenafil", "penegra": "sildenafil", "manforce": "sildenafil",
    "flomax": "tamsulosin", "urimax": "tamsulosin",
}

# Metadata notes for common combination / brand formulations
BRAND_NOTES = {
    "combiflam": "Combiflam is a fixed-dose combination of Ibuprofen (400mg) and Paracetamol (325mg). Checked against Ibuprofen NSAID interactions.",
    "montair-lc": "Montair-LC is a combination of Montelukast and Levocetirizine. Checked against Montelukast FDA label.",
    "pan-d": "Pan-D contains Pantoprazole and Domperidone. Checked against Pantoprazole FDA label.",
    "augmentin": "Augmentin contains Amoxicillin and Clavulanic Acid. Checked against Amoxicillin FDA label.",
    "ecosprin": "Ecosprin is an Indian brand of enteric-coated low-dose Aspirin (75mg/150mg).",
    "dolo 650": "Dolo-650 is an Indian formulation of Paracetamol (Acetaminophen 650mg).",
}


def detect_drugs(text):
    """Return the generic drug names mentioned in a question, in order of appearance."""
    t = text.lower()
    # Sort candidate names by descending length so multi-word aliases match before substrings
    candidates = sorted(list(DRUGS) + list(ALIASES.keys()), key=len, reverse=True)
    hits = []
    matched_spans = []

    for name in candidates:
        for m in re.finditer(rf"\b{re.escape(name)}\b", t):
            start, end = m.span()
            # Avoid overlapping sub-matches (e.g. 'dolo' inside 'dolo 650')
            if any(s <= start < e or s < end <= e for s, e in matched_spans):
                continue
            matched_spans.append((start, end))
            generic = ALIASES.get(name, name)
            hits.append((start, generic))

    hits.sort(key=lambda x: x[0])
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
