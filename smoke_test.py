"""Offline smoke test. Needs no internet, API key, or model download.
Run it right after installing to confirm your library versions work with this code:
    python smoke_test.py
"""
import json
import os
import tempfile

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import build_index
import eval_retrieval
import rag
from drugs import detect_drugs

# 1. drug detection
assert detect_drugs("Can I take ibuprofen with warfarin?") == ["ibuprofen", "warfarin"]
assert detect_drugs("Is Tylenol safe?") == ["acetaminophen"]
assert detect_drugs("What is the weather?") == []

# 2. build a tiny index with fake embeddings
tmp = tempfile.mkdtemp()
labels = os.path.join(tmp, "labels.jsonl")
rows = [
    {"drug": "ibuprofen", "section": "Indications and Usage",
     "text": "Ibuprofen is indicated for relief of mild to moderate pain and fever. " * 6, "set_id": "a"},
    {"drug": "warfarin", "section": "Drug Interactions",
     "text": "Aspirin and NSAIDs such as ibuprofen increase bleeding risk with warfarin. " * 6, "set_id": "b"},
    {"drug": "lisinopril", "section": "Contraindications",
     "text": "Lisinopril is contraindicated in patients with a history of angioedema. " * 6, "set_id": "c"},
]
with open(labels, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

rag.get_embeddings = lambda: DeterministicFakeEmbedding(size=64)
rag.get_llm = lambda: FakeListChatModel(responses=["ok [warfarin | Drug Interactions]"])
db_dir = os.path.join(tmp, "db")
n = build_index.build(chunk_size=200, overlap=20, out_dir=db_dir, labels_path=labels)
assert n >= 3

# 3. retrieval: single drug, two drugs, unknown drug, explicit drug
docs = rag.retrieve("What are the contraindications of lisinopril?", persist_dir=db_dir)
assert docs and all(d.metadata["drug"] == "lisinopril" for d in docs)
docs = rag.retrieve("Can I take ibuprofen with warfarin?", persist_dir=db_dir)
assert {"ibuprofen", "warfarin"} <= {d.metadata["drug"] for d in docs}
docs = rag.retrieve("Tell me about metformin", persist_dir=db_dir)  # not indexed -> fallback
assert docs
docs = rag.retrieve("anything", drug="warfarin", persist_dir=db_dir)
assert all(d.metadata["drug"] == "warfarin" for d in docs)

# 4. full chain
text, docs = rag.answer("Can I take ibuprofen with warfarin?", persist_dir=db_dir)
assert text.startswith("ok") and docs

# 5. eval helper
data = [{"question": "What are the contraindications of lisinopril?", "drug": "lisinopril", "section": "Contraindications"}]
assert eval_retrieval.hit_rate(data, 4, True, db_dir) == 1.0

print("SMOKE TEST PASSED")
