"""Regression tests for interaction verification, multi-section retrieval, and safety post-check.

Validates the general remediation of Case #19 (simvastatin + amlodipine):
- Multi-section safety retrieval across Drug Interactions, Dosage, Warnings, Contraindications
- Unmentioned interactions are treated as 'could not verify' and never 'safe'
- Safety post-check prevents and overrides false reassurance in agent drafts
"""
import json
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import agent
import tools

LABEL_ROWS = [
    {
        "drug": "simvastatin",
        "section": "Dosage and Administration",
        "set_id": "simva_1",
        "text": "Patients taking Amiodarone, Amlodipine, or Ranolazine: Do not exceed simvastatin tablets 20 mg once daily."
    },
    {
        "drug": "simvastatin",
        "section": "Drug Interactions",
        "set_id": "simva_1",
        "text": "See Dosage and Administration for details regarding concomitant use that increases myopathy and rhabdomyolysis risk."
    },
    {
        "drug": "amlodipine",
        "section": "Drug Interactions",
        "set_id": "amlo_1",
        "text": "Co-administration of multiple doses of 10 mg of amlodipine with 80 mg simvastatin resulted in an increase in exposure to simvastatin."
    },
    {
        "drug": "metformin",
        "section": "Drug Interactions",
        "set_id": "met_1",
        "text": "Carbonic anhydrase inhibitors may increase the risk of lactic acidosis."
    },
    {
        "drug": "lisinopril",
        "section": "Drug Interactions",
        "set_id": "lis_1",
        "text": "Dual inhibition of the renin-angiotensin system increases risk of hypotension, syncope, and hyperkalemia."
    }
]


@pytest.fixture()
def regression_env(tmp_path, monkeypatch):
    labels_path = tmp_path / "labels.jsonl"
    labels_path.write_text("\n".join(json.dumps(r) for r in LABEL_ROWS), encoding="utf-8")
    monkeypatch.setattr(tools, "LABELS", str(labels_path))
    return str(labels_path)


def test_simvastatin_amlodipine_dosage_ceiling(regression_env):
    """Regression test for Case #19: verifies that simvastatin + amlodipine flags the 20mg dosage limit

    from the Dosage and Administration section.
    """
    out = tools.check_interaction.invoke({"drug_a": "simvastatin", "drug_b": "amlodipine"})
    assert "at least one label flags this pair" in out
    assert "[simvastatin | Dosage and Administration] mentions 'amlodipine'" in out
    assert "20 mg" in out


def test_unverified_interaction_is_never_called_safe(regression_env):
    """Verifies that an unmentioned pair returns 'could not verify' and explicit 'NOT proof of safety'."""
    out = tools.check_interaction.invoke({"drug_a": "metformin", "drug_b": "lisinopril"})
    assert "could not verify interaction" in out
    assert "NOT proof of safety" in out
    assert "safe to take" not in out.lower()


def test_multidrug_retrieval_scans_all_safety_sections(regression_env):
    """Verifies that check_medication_list detects interactions across non-interactions sections."""
    out = tools.check_medication_list.invoke({"medications": "simvastatin, amlodipine, metformin"})
    assert "Polypharmacy Interaction Audit" in out
    assert "Simvastatin + Amlodipine" in out
    assert "Dosage and Administration" in out
    assert "Could Not Verify" in out or "absence of an explicit warning" in out


def test_postcheck_replaces_false_reassurance_on_flagged_pair():
    """Verifies that agent.post_check_safety overrides false reassurance when an interaction is flagged."""
    draft = "It is safe to take simvastatin with amlodipine without any issues."
    steps = [{
        "tool": "check_interaction",
        "output": "[simvastatin | Dosage and Administration] mentions 'amlodipine'... at least one label flags this pair."
    }]
    corrected = agent.post_check_safety(draft, steps)
    assert "Warning: A potential interaction or dosage limitation was identified" in corrected
    assert "Please do not assume this combination is safe" in corrected


def test_postcheck_replaces_false_reassurance_on_unverified_pair():
    """Verifies that agent.post_check_safety overrides false reassurance when an interaction is unverified."""
    draft = "There are no known interactions between metformin and lisinopril, so it is safe to take together."
    steps = [{
        "tool": "check_interaction",
        "output": "could not verify interaction in official package insert sections. This is NOT proof of safety."
    }]
    corrected = agent.post_check_safety(draft, steps)
    assert "Caution: Interaction data between these medications could not be verified" in corrected
    assert "Absence of an explicit warning in a drug label is NOT proof of safety" in corrected

