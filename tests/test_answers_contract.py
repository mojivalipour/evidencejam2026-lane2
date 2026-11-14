"""Output contract: status consistency and deterministic citation checks."""

import json

import pytest
from pydantic import ValidationError

from causa_nostra.answers.models import AnswerEntry, AnswersFile
from causa_nostra.answers.rubric import (
    ANSWERED_QUESTIONS,
    ANSWERED_SCORED_CATEGORIES,
    ANSWERED_WEIGHTS,
    NO_EVIDENCE_QUESTIONS,
    NO_EVIDENCE_WEIGHTS,
    weighted_score,
)
from causa_nostra.answers.validate import check_file, validate_answer_file
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


def test_validate_answer_file_checks_fields_without_loading_graph(tmp_path):
    path = tmp_path / "answers.json"
    answers = AnswersFile(answers=[AnswerEntry(**_answered(edge_ids=["not-in-graph"]))])
    path.write_text(answers.model_dump_json(), encoding="utf-8")

    assert validate_answer_file(path) == []


def test_validate_answer_file_reports_invalid_answer_fields(tmp_path):
    path = tmp_path / "answers.json"
    invalid = {
        "team": "CAUSA NOSTRA",
        "answers": [_no_evidence(confidence=1.5, unexpected_field="not allowed")],
    }
    path.write_text(json.dumps(invalid), encoding="utf-8")

    problems = validate_answer_file(path)

    assert any("confidence" in problem for problem in problems)
    assert any("unexpected_field" in problem for problem in problems)


def test_rubric_weights_sum_to_one_and_score_deterministically():
    assert round(sum(ANSWERED_WEIGHTS.values()), 6) == 1.0
    assert round(sum(NO_EVIDENCE_WEIGHTS.values()), 6) == 1.0
    question_ids = {
        question_id
        for category in ANSWERED_SCORED_CATEGORIES
        for question_id in ANSWERED_QUESTIONS[category]
    }
    all_pass = {question_id: "pass" for question_id in question_ids}
    all_irrelevant = {question_id: "irrelevant" for question_id in question_ids}
    all_fail = {question_id: "fail" for question_id in question_ids}

    assert weighted_score(
        all_pass, ANSWERED_WEIGHTS, ANSWERED_QUESTIONS, ANSWERED_SCORED_CATEGORIES
    ) == 1.0
    assert weighted_score(
        all_irrelevant,
        ANSWERED_WEIGHTS,
        ANSWERED_QUESTIONS,
        ANSWERED_SCORED_CATEGORIES,
    ) == 0.5
    assert weighted_score(
        all_fail, ANSWERED_WEIGHTS, ANSWERED_QUESTIONS, ANSWERED_SCORED_CATEGORIES
    ) == 0.0
    no_evidence_ids = {
        question_id
        for category in ("interpretation", "no_evidence_justification")
        for question_id in NO_EVIDENCE_QUESTIONS[category]
    }
    assert weighted_score(
        {question_id: "pass" for question_id in no_evidence_ids},
        NO_EVIDENCE_WEIGHTS,
        NO_EVIDENCE_QUESTIONS,
        ("interpretation", "no_evidence_justification"),
    ) == 1.0
