"""A failed first search gets one broader search before the question is called no_evidence."""

from __future__ import annotations

from pathlib import Path

from causa_nostra.answers.models import AnswerEntry
from causa_nostra.graph.store import GraphStore
from causa_nostra.pipeline.run import answer_question
from causa_nostra.pipeline.stages import Stages
from causa_nostra.pipeline.types import Question, QuestionFrame, Verification

GRAPH = Path(__file__).resolve().parents[1] / "causal_graph.json"
QUESTION = Question(id="T-1", question="What does burning do to non-native grass cover?")


def _frame(terms: list[str] | None = None) -> QuestionFrame:
    return QuestionFrame(
        question_id="T-1",
        intervention="burning",
        outcome="grass cover",
        intervention_terms=terms or [],
    )


def _no_evidence(question: Question, reason_count: list[int]) -> AnswerEntry:
    return AnswerEntry(
        id=question.id,
        status="no_evidence",
        answer="The graph has no evidence answering this intervention–outcome question.",
        direction=None,
        confidence=0.5,
        certainty=None,
        edge_ids=[],
        searched_for="fake search",
    )


def _stages(refine_result: QuestionFrame | None, *, refine_enabled: bool = True) -> tuple[Stages, dict[str, int]]:
    calls = {"retrieve": 0, "reason": 0, "refine": 0}

    def understand(question: Question) -> QuestionFrame:
        return _frame()

    def retrieve(frame: QuestionFrame) -> list:
        calls["retrieve"] += 1
        return []

    def reason(question, frame, records, feedback) -> AnswerEntry:
        calls["reason"] += 1
        return _no_evidence(question, [])

    def verify(question, frame, draft, records) -> Verification:
        return Verification(passed=True, score=0.8)

    def refine(question: Question, frame: QuestionFrame) -> QuestionFrame:
        calls["refine"] += 1
        return refine_result if refine_result is not None else frame

    stages = Stages(
        understand=understand,
        retrieve=retrieve,
        reason=reason,
        verify=verify,
        refine=refine if refine_enabled else None,
    )
    return stages, calls


def test_no_evidence_triggers_one_broader_search() -> None:
    store = GraphStore.from_json(GRAPH)
    stages, calls = _stages(_frame(["controlled fire"]))

    answer_question(QUESTION, store, stages)

    assert calls["refine"] == 1
    assert calls["retrieve"] == 2  # original search, then the broader one
    assert calls["reason"] == 2


def test_no_new_terms_skips_second_search() -> None:
    store = GraphStore.from_json(GRAPH)
    stages, calls = _stages(None)

    answer_question(QUESTION, store, stages)

    assert calls["refine"] == 1
    assert calls["retrieve"] == 1
    assert calls["reason"] == 1


def test_refinement_disabled_when_stage_missing() -> None:
    store = GraphStore.from_json(GRAPH)
    stages, calls = _stages(_frame(["controlled fire"]), refine_enabled=False)

    answer_question(QUESTION, store, stages)

    assert calls["refine"] == 0
    assert calls["retrieve"] == 1


def test_merge_drops_case_insensitive_duplicates_and_caps() -> None:
    from causa_nostra.agents.question_processor import _merge

    assert _merge(["Fire"], ["fire", "burn"]) == ["Fire", "burn"]
    assert len(_merge([f"t{i}" for i in range(30)], [])) == 20
