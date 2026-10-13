"""Offline tests: no internet, API key or model download needed."""
import json
import os
import sys

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import agent  # noqa: E402
import build_index  # noqa: E402
import rag  # noqa: E402
import tools  # noqa: E402

ROWS = [
    {"drug": "ibuprofen", "section": "Drug Interactions", "set_id": "a",
     "text": "Ibuprofen may increase bleeding risk when given with warfarin. " * 5},
    {"drug": "warfarin", "section": "Drug Interactions", "set_id": "b",
     "text": "Drugs that increase bleeding risk include NSAIDs and aspirin. " * 5},
    {"drug": "lisinopril", "section": "Contraindications", "set_id": "c",
     "text": "Lisinopril is contraindicated in patients with a history of angioedema. " * 5},
    {"drug": "lisinopril", "section": "Drug Interactions", "set_id": "c",
     "text": "Lithium levels may rise with lisinopril. " * 5},
    {"drug": "metformin", "section": "Drug Interactions", "set_id": "d",
     "text": "Carbonic anhydrase inhibitors may increase lactic acidosis risk. " * 5},
]


class Scripted(BaseChatModel):
    script: list
    i: int = 0

    @property
    def _llm_type(self):
        return "scripted"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        item = self.script[self.i]
        self.i += 1
        if isinstance(item, Exception):
            raise item
        return ChatResult(generations=[ChatGeneration(message=item)])


@pytest.fixture()
def env(tmp_path, monkeypatch):
    labels = tmp_path / "labels.jsonl"
    labels.write_text("\n".join(json.dumps(r) for r in ROWS), encoding="utf-8")
    monkeypatch.setattr(rag, "get_embeddings", lambda: DeterministicFakeEmbedding(size=64))
    db = str(tmp_path / "db")
    build_index.build(chunk_size=200, overlap=20, out_dir=db, labels_path=str(labels))
    monkeypatch.setattr(tools, "DB_DIR", db)
    monkeypatch.setattr(tools, "LABELS", str(labels))
    return db


def test_interaction_flagged(env):
    out = tools.check_interaction.invoke({"drug_a": "Advil", "drug_b": "warfarin"})  # brand name works
    assert "[ibuprofen | Drug Interactions] mentions" in out
    assert "[warfarin | Drug Interactions] mentions 'nsaids'" in out
    assert "at least one label flags" in out


def test_interaction_not_found_is_not_called_safe(env):
    out = tools.check_interaction.invoke({"drug_a": "lisinopril", "drug_b": "metformin"})
    assert "NOT proof of safety" in out


def test_interaction_unknown_drug(env):
    assert "Not in the index: foodrug" in tools.check_interaction.invoke({"drug_a": "foodrug", "drug_b": "warfarin"})


def test_search_labels_filters_by_drug(env):
    out = tools.search_labels.invoke({"query": "angioedema", "drug": "lisinopril"})
    assert out.startswith("[lisinopril |")


def test_faers_parsing_and_failure(monkeypatch):
    class R:
        def raise_for_status(self): pass
        def json(self): return {"results": [{"term": "NAUSEA", "count": 10}, {"term": "RASH", "count": 4}]}
    monkeypatch.setattr(tools.requests, "get", lambda *a, **k: R())
    out = tools.adverse_event_stats.invoke({"drug": "ibuprofen"})
    assert "NAUSEA (10)" in out and "do not prove" in out

    def boom(*a, **k):
        raise tools.requests.ConnectionError()
    monkeypatch.setattr(tools.requests, "get", boom)
    assert "unavailable" in tools.adverse_event_stats.invoke({"drug": "ibuprofen"})


def test_agent_calls_tool_then_answers(env):
    call = AIMessage(content="", tool_calls=[{"name": "check_interaction", "id": "c1",
                                              "args": {"drug_a": "ibuprofen", "drug_b": "warfarin"}}])
    final = AIMessage(content="Yes, flagged [warfarin | Drug Interactions].")
    text, steps = agent.ask(agent.make_graph(Scripted(script=[call, final])), "Can I take ibuprofen with warfarin?")
    assert text.startswith("Yes") and [s["tool"] for s in steps] == ["check_interaction"]
    assert "mentions" in steps[0]["output"]


def test_emergency_skips_the_llm():
    model = Scripted(script=[])   # would raise IndexError if the model were called
    text, steps = agent.ask(agent.make_graph(model), "My friend took an overdose of paracetamol")
    assert "emergency" in text.lower() and steps == []


def test_llm_failure_is_handled():
    model = Scripted(script=[RuntimeError("x"), RuntimeError("x")])
    text, _ = agent.ask(agent.make_graph(model), "What is metformin for?")
    assert "failed" in text
