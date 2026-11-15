"""Multi-part questions: facet extraction, per-facet retrieval, and the no-evidence guard. No network calls."""

import json

import pytest

from causa_nostra.agents import QuestionProcessor
from causa_nostra.answers.models import AnswerEntry
from causa_nostra.graph.models import Graph, Node
from causa_nostra.graph.store import GraphStore
from causa_nostra.llm.client import ChatResult
from causa_nostra.pipeline.run import NO_EVIDENCE_WITH_CANDIDATES_CAP, answer_question
from causa_nostra.pipeline.stages import Stages
from causa_nostra.pipeline.types import Facet, Question, QuestionFrame, Verification
from causa_nostra.retrieval.pipeline import RetrievalPipeline, _round_robin, search_query
from causa_nostra.retrieval.records import EvidenceRecord


class FakeClient:
    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)

    def chat(self, messages, *, temperature=0.0, max_tokens=None, thinking=True):
        return ChatResult(text=self.replies.pop(0))


def _record(edge_id: str, subject: str, obj: str) -> EvidenceRecord:
    return EvidenceRecord(
        edge_id=edge_id, subject_id="S", object_id="O", subject_label=subject, object_label=obj,
        subject_qualifier=None, object_qualifier=None, predicate="causes", causal_language="direct_causal",
        negated=False, relation_sign="positive", source_document="d1", original_sentence="",
        ecosystem_context=None, certainty_grade=None, text=f"subject: {subject} | object: {obj}",
    )


QUESTION = Question(id="M-1", question="What do contaminants such as PCBs and oil sands tailings do to wildlife?")


# --- facet extraction -------------------------------------------------------

def test_question_processor_parses_facets_and_skips_bad_entries():
    reply = json.dumps({
        "intervention": "contaminants", "outcome": "wildlife effects",
        "intervention_terms": [], "outcome_terms": [],
        "facets": [
            {"intervention": "PCBs", "outcome": "crab PCB concentration", "intervention_terms": ["PCB"]},
            {"intervention": "oil sands tailings", "outcome": "fledging success"},
            {"intervention": "PCBs", "outcome": "crab PCB concentration"},  # duplicate
            {"intervention": "", "outcome": "x"},  # incomplete
            "not an object",
        ],
    })
    frame = QuestionProcessor(FakeClient(reply)).run(QUESTION)
    assert [(f.intervention, f.outcome) for f in frame.facets] == [
        ("PCBs", "crab PCB concentration"),
        ("oil sands tailings", "fledging success"),
    ]
    assert frame.facets[0].intervention_terms == ["PCB"]


def test_question_processor_without_facets_gives_empty_list():
    reply = json.dumps({"intervention": "burning", "outcome": "grass cover"})
    assert QuestionProcessor(FakeClient(reply)).run(QUESTION).facets == []


# --- retrieval --------------------------------------------------------------

def test_round_robin_interleaves_and_dedupes():
    assert _round_robin([["a", "b", "c"], ["x", "a"], ["y"]], limit=10) == ["a", "x", "y", "b", "c"]
    assert _round_robin([["a", "b"], ["x", "y"]], limit=3) == ["a", "x", "b"]


def test_facets_recover_edges_the_umbrella_query_misses():
    records = [
        _record("e_generic", "toxic contaminant effects", "lethal effects on wildlife"),
        _record("e_pcb", "PCB remediation", "PCB concentration in rock crab"),
        _record("e_oil", "oil sands process material", "fledging success of tree swallows"),
    ]
    pipeline = RetrievalPipeline(records, top_k=5)
    umbrella = QuestionFrame(question_id="M-1", intervention="toxic contaminants", outcome="lethal effects wildlife")
    assert [r.edge_id for r in pipeline.retrieve(umbrella)] == ["e_generic"]

    faceted = umbrella.model_copy(update={"facets": [
        Facet(intervention="PCB", outcome="PCB concentration crab"),
        Facet(intervention="oil sands", outcome="fledging success"),
    ]})
    assert {r.edge_id for r in pipeline.retrieve(faceted)} == {"e_generic", "e_pcb", "e_oil"}
    assert "facet: subject side: [PCB]" in search_query(faceted)


# --- no-evidence guard ------------------------------------------------------

STORE = GraphStore(Graph(graph_id="t", nodes=[Node(id="S"), Node(id="O")], edges=[]))
RECORDS = [_record("e1", "PCB", "crab")]


def _no_evidence(qid: str) -> AnswerEntry:
    return AnswerEntry(id=qid, status="no_evidence", answer="none", confidence=0.9, searched_for="PCB crab")


def _stages(reason, retrieve=lambda f: RECORDS, score=1.0) -> Stages:
    return Stages(
        understand=lambda q: QuestionFrame(question_id=q.id, intervention="PCB", outcome="crab"),
        retrieve=retrieve,
        reason=reason,
        verify=lambda q, f, d, r: Verification(passed=True, score=score),
    )


def test_no_evidence_with_candidates_is_rechecked_then_capped():
    feedback_seen: list[list[str]] = []

    def reason(q, f, r, feedback):
        feedback_seen.append(feedback)
        return _no_evidence(q.id)

    entry = answer_question(QUESTION, STORE, _stages(reason))
    assert entry.status == "no_evidence"
    assert entry.confidence == pytest.approx(NO_EVIDENCE_WITH_CANDIDATES_CAP)
    assert len(feedback_seen) == 2 and "no_evidence" in feedback_seen[1][0]


def test_recheck_can_turn_no_evidence_into_an_answer():
    def reason(q, f, r, feedback):
        if not feedback:
            return _no_evidence(q.id)
        return AnswerEntry(id=q.id, status="answered", answer="PCB rose [e1].", direction="increase",
                           confidence=0.6, certainty="low", edge_ids=["e1"])

    store = GraphStore(Graph.model_validate({
        "graph_id": "t", "nodes": [{"id": "S"}, {"id": "O"}],
        "edges": [{"id": "e1", "subject": "S", "predicate": "causes", "object": "O", "causal_language": "direct_causal"}],
    }))
    entry = answer_question(QUESTION, store, _stages(reason, score=0.8))
    assert entry.status == "answered" and entry.confidence == pytest.approx(0.8)


def test_no_evidence_without_candidates_keeps_rubric_confidence():
    entry = answer_question(QUESTION, STORE, _stages(lambda q, f, r, fb: _no_evidence(q.id), retrieve=lambda f: []))
    assert entry.confidence == pytest.approx(1.0)


def test_failed_recheck_falls_back_to_capped_no_evidence():
    calls = {"n": 0}

    def reason(q, f, r, feedback):
        calls["n"] += 1
        if calls["n"] == 1:
            return _no_evidence(q.id)
        # Hallucinated citation: fails the deterministic check on the last attempt.
        return AnswerEntry(id=q.id, status="answered", answer="x", direction="increase",
                           confidence=0.6, certainty="low", edge_ids=["missing"])

    entry = answer_question(QUESTION, STORE, _stages(reason))
    assert entry.status == "no_evidence"
    assert entry.confidence == pytest.approx(NO_EVIDENCE_WITH_CANDIDATES_CAP)
