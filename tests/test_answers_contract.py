"""Output contract: status consistency and deterministic citation checks."""

import pytest
from pydantic import ValidationError

from causa_nostra.answers.models import AnswerEntry, AnswersFile
from causa_nostra.answers.rubric import ANSWERED_WEIGHTS, NO_EVIDENCE_WEIGHTS, weighted_score
from causa_nostra.answers.validate import check_file
from causa_nostra.graph.models import Edge, Graph, Node
from causa_nostra.graph.store import GraphStore


def _answered(**overrides) -> dict:
    base = {
        "id": "Q1",
        "status": "answered",
        "answer": "Probably yes.",
        "direction": "decrease",
        "confidence": 0.7,
        "certainty": "moderate",
        "edge_ids": ["e1"],
    }
    base.update(overrides)
    return base


def _no_evidence(**overrides) -> dict:
    base = {
        "id": "Q2",
        "status": "no_evidence",
        "answer": "No evidence in the graph.",
        "direction": None,
        "confidence": 0.8,
        "certainty": None,
        "edge_ids": [],
        "searched_for": "edges from mowing to soil carbon",
    }
    base.update(overrides)
    return base


def test_valid_entries_pass():
    AnswerEntry(**_answered())
    AnswerEntry(**_no_evidence())


def test_answered_requires_direction_and_edges():
    with pytest.raises(ValidationError):
        AnswerEntry(**_answered(direction=None))
    with pytest.raises(ValidationError):
        AnswerEntry(**_answered(edge_ids=[]))


def test_no_evidence_must_not_cite_or_grade():
    with pytest.raises(ValidationError):
        AnswerEntry(**_no_evidence(edge_ids=["e1"]))
    with pytest.raises(ValidationError):
        AnswerEntry(**_no_evidence(certainty="low"))
    with pytest.raises(ValidationError):
        AnswerEntry(**_no_evidence(searched_for=None))


def test_confidence_bounds():
    with pytest.raises(ValidationError):
        AnswerEntry(**_answered(confidence=1.5))


def test_file_rejects_duplicate_ids():
    with pytest.raises(ValidationError):
        AnswersFile(answers=[AnswerEntry(**_answered()), AnswerEntry(**_answered())])


def test_check_file_reports_missing_unknown_and_bad_citations():
    store = GraphStore(
        Graph(
            graph_id="t",
            nodes=[Node(id="S"), Node(id="O")],
            edges=[Edge(id="e1", subject="S", predicate="causes", object="O", causal_language="direct_causal")],
        )
    )
    answers = AnswersFile(answers=[AnswerEntry(**_answered(edge_ids=["e1", "missing"]))])
    problems = check_file(answers, store, ["Q1", "Q3"])
    assert any("missing answer for question Q3" in p for p in problems)
    assert any("citation missing does not exist" in p for p in problems)


def test_rubric_weights_sum_to_one_and_score_deterministically():
    assert round(sum(ANSWERED_WEIGHTS.values()), 6) == 1.0
    assert round(sum(NO_EVIDENCE_WEIGHTS.values()), 6) == 1.0
    full = weighted_score({k: "pass" for k in ANSWERED_WEIGHTS}, ANSWERED_WEIGHTS)
    half = weighted_score({k: "partial" for k in ANSWERED_WEIGHTS}, ANSWERED_WEIGHTS)
    assert full == 1.0 and half == 0.5
