"""Rubric tests: rubric.py is the single source of truth for scoring and gates."""

from __future__ import annotations

import pytest

from causa_nostra.agents.verifier import SYSTEM_PROMPT
from causa_nostra.answers.rubric import (
    ANSWERED_GATE_QUESTIONS,
    ANSWERED_QUESTIONS,
    ANSWERED_WEIGHTS,
    NO_EVIDENCE_GATE_QUESTIONS,
    evaluate_answered,
    evaluate_no_evidence,
    normalise,
    rubric_score,
)


def _all(judgement: str, questions: dict[str, dict[str, str]]) -> dict[str, str]:
    return {item: judgement for category in questions.values() for item in category}


def test_all_pass_scores_one_and_passes() -> None:
    result = evaluate_answered(_all("pass", ANSWERED_QUESTIONS))
    assert result.score == pytest.approx(1.0)
    assert result.passed


def test_partial_sub_item_downgrades_category_only() -> None:
    items = _all("pass", ANSWERED_QUESTIONS) | {"D4": "partial"}
    result = evaluate_answered(items)
    assert result.criteria["search_coverage"] == "partial"
    assert result.score == pytest.approx(0.925)
    assert result.passed


def test_failed_citation_gates_even_with_high_score() -> None:
    items = _all("pass", ANSWERED_QUESTIONS) | {"C1": "fail"}
    result = evaluate_answered(items)
    assert result.score > 0.7
    assert "C1" in result.failed_gates
    assert not result.passed


def test_failed_search_item_does_not_gate() -> None:
    items = _all("pass", ANSWERED_QUESTIONS) | {"D1": "fail"}
    result = evaluate_answered(items)
    assert result.criteria["search_coverage"] == "fail"
    assert result.passed


def test_missing_and_invalid_items_count_as_fail() -> None:
    items = _all("pass", ANSWERED_QUESTIONS)
    del items["B2"]
    items["A1"] = "maybe"
    result = evaluate_answered(items)
    assert result.items["B2"] == "fail"
    assert result.items["A1"] == "fail"
    assert not result.passed


def test_normalise_lowercases_and_rejects_unknown() -> None:
    assert normalise(" PASS ") == "pass"
    assert normalise("yes") == "fail"


def test_irrelevant_item_is_not_scored_or_gated() -> None:
    items = _all("pass", ANSWERED_QUESTIONS) | {"A4": "irrelevant"}
    result = evaluate_answered(items)
    assert result.items["A4"] == "irrelevant"
    assert result.criteria["evidence_match"] == "pass"
    assert result.score == pytest.approx(1.0)
    assert result.passed


@pytest.mark.parametrize("item", [i for c in ANSWERED_QUESTIONS.values() for i in c if i != "A4"])
def test_mandatory_item_cannot_be_irrelevant(item: str) -> None:
    # Regression: irrelevant used to bypass mandatory checks such as A1 (intervention match).
    result = evaluate_answered(_all("pass", ANSWERED_QUESTIONS) | {item: "irrelevant"})
    assert result.items[item] == "fail"
    assert result.passed == (item not in ANSWERED_GATE_QUESTIONS)


def test_intervention_mismatch_cannot_bypass_as_irrelevant() -> None:
    result = evaluate_answered(_all("pass", ANSWERED_QUESTIONS) | {"A1": "irrelevant"})
    assert result.score < 1.0
    assert result.passed is False
    assert result.failed_gates == ("A1",)


def test_citation_item_cannot_be_irrelevant() -> None:
    items = _all("pass", ANSWERED_QUESTIONS) | {"C2": "irrelevant"}
    result = evaluate_answered(items)
    assert result.items["C2"] == "fail"
    assert "C2" in result.failed_gates
    assert not result.passed


def test_unknown_word_is_still_fail() -> None:
    items = _all("pass", ANSWERED_QUESTIONS) | {"A1": "not applicable"}
    assert evaluate_answered(items).items["A1"] == "fail"


def test_rubric_score_missing_categories_count_as_fail() -> None:
    assert rubric_score({name: "pass" for name in ANSWERED_WEIGHTS}) == pytest.approx(1.0)
    assert rubric_score({}) == pytest.approx(0.0)


def test_no_evidence_gate_on_absence_claim() -> None:
    items = {"N1": "pass", "N2": "pass", "N3": "pass", "N4": "pass", "N5": "pass", "N6": "fail"}
    result = evaluate_no_evidence(items)
    assert result.score == pytest.approx(1.0)
    assert result.failed_gates == ("N6",)
    assert not result.passed


def test_prompt_contains_every_answered_sub_item() -> None:
    for item in (item for category in ANSWERED_QUESTIONS.values() for item in category):
        assert f'"{item}"' in SYSTEM_PROMPT


def test_weighted_score_rejects_invalid_judgement_and_missing_criteria() -> None:
    from causa_nostra.answers.rubric import weighted_score

    valid = {name: "pass" for name in ANSWERED_WEIGHTS}
    with pytest.raises(ValueError):
        weighted_score(valid | {"evidence_match": "maybe"}, ANSWERED_WEIGHTS)
    with pytest.raises(ValueError):
        weighted_score({"evidence_match": "pass"}, ANSWERED_WEIGHTS)


def test_no_evidence_partial_item_is_weighted() -> None:
    items = {"N1": "pass", "N2": "pass", "N3": "pass", "N4": "partial", "N5": "pass", "N6": "pass"}
    result = evaluate_no_evidence(items)
    assert result.criteria["candidates_checked"] == "partial"
    assert result.score == pytest.approx(0.85)
    assert result.passed


def test_no_evidence_gate_irrelevant_is_a_failure() -> None:
    # Only explicitly optional items may be irrelevant; N6 is mandatory.
    items = {"N1": "pass", "N2": "pass", "N3": "pass", "N4": "pass", "N5": "pass", "N6": "irrelevant"}
    result = evaluate_no_evidence(items)
    assert result.items["N6"] == "fail"
    assert result.passed is False


@pytest.mark.parametrize("item", ["N1", "N2", "N3", "N4", "N5", "N6"])
def test_every_no_evidence_item_gates_acceptance(item: str) -> None:
    # Regression: only N6 gated, so N5=fail (evidence was found) still passed.
    items = {n: "pass" for n in ["N1", "N2", "N3", "N4", "N5", "N6"]} | {item: "fail"}
    result = evaluate_no_evidence(items)
    assert result.passed is False
    assert item in result.failed_gates


def test_no_evidence_missing_items_block_acceptance() -> None:
    # Regression: omitting N1-N5 used to pass with score 0.0.
    result = evaluate_no_evidence({"N6": "pass"})
    assert result.passed is False
    assert result.score == pytest.approx(0.0)
