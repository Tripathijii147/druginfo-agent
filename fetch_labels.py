"""Step 1: download FDA drug labels from openFDA and save one record per label section."""
import json
import os
import re
import time

import requests

from drugs import DRUGS

URL = "https://api.fda.gov/drug/label.json"
API_KEY = os.getenv("OPENFDA_API_KEY")  # optional, raises the rate limit
MAX_CHARS_PER_SECTION = 6000            # keeps the index small and fast

SECTIONS = {
    "indications_and_usage": "Indications and Usage",
    "dosage_and_administration": "Dosage and Administration",
    "contraindications": "Contraindications",
    "warnings_and_cautions": "Warnings and Precautions",
    "warnings": "Warnings",
    "drug_interactions": "Drug Interactions",
    "adverse_reactions": "Adverse Reactions",
}


def score(rec):
    """Prefer the label that has the most sections we care about."""
    return sum(1 for key in SECTIONS if rec.get(key))


def fetch(drug):
    queries = (
        f'openfda.generic_name.exact:"{drug.upper()}"',  # single-ingredient labels only
        f'openfda.generic_name:"{drug}"',                # looser fallback
    )
    for query in queries:
        params = {"search": query, "limit": 10}
        if API_KEY:
            params["api_key"] = API_KEY
        try:
            r = requests.get(URL, params=params, timeout=30)
        except requests.RequestException as e:
            print(f"  network error for {drug}: {e}")
            continue
        if r.status_code != 200:
            continue
        results = r.json().get("results", [])
        if results:
            return max(results, key=score)
    return None


def clean(text):
    return re.sub(r"\s+", " ", text).strip()


def main():
    os.makedirs("data", exist_ok=True)
    kept, missing = 0, []
    with open("data/labels.jsonl", "w", encoding="utf-8") as f:
        for drug in DRUGS:
            rec = fetch(drug)
            if not rec:
                missing.append(drug)
                print(f"no label found: {drug}")
                continue
            n = 0
            for key, title in SECTIONS.items():
                texts = rec.get(key)
                if not texts:
                    continue
                text = clean(" ".join(texts))[:MAX_CHARS_PER_SECTION]
                f.write(json.dumps({"drug": drug, "section": title, "text": text,
                                    "set_id": rec.get("set_id") or ""}) + "\n")
                n += 1
            kept += n
            print(f"{drug}: {n} sections")
            time.sleep(0.3)
    print(f"\nSaved {kept} sections to data/labels.jsonl. Missing drugs: {missing or 'none'}")


if __name__ == "__main__":
    main()
