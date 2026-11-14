"""Every input question gets an entry, even when a draft cannot be verified."""

from pathlib import Path

from causa_nostra.answers.models import AnswerEntry
from causa_nostra.answers.validate import check_entry, check_extras
from causa_nostra.graph.models import Edge, Graph, Node
from causa_nostra.graph.store import GraphStore
from causa_nostra.pipeline.run import run_pipeline
from causa_nostra.pipeline.stages import Stages
from causa_nostra.pipeline.types import Question, QuestionFrame, Verification


def _store() -> GraphStore:
    return GraphStore(
        Graph(
            graph_id="t",
            nodes=[Node(id="S"), Node(id="O")],
            edges=[Edge(id="e1", subject="S", predicate="causes", object="O", causal_language="direct_causal")],
        )
    )


def _stages(reason) -> Stages:
    return Stages(
        understand=lambda q: QuestionFrame(question_id=q.id, intervention="S", outcome="O"),
        retrieve=lambda frame: [],
        reason=reason,
        verify=lambda q, d, r: Verification(passed=True),
    )


def _bad_citation_reason(question, frame, records, feedback) -> AnswerEntry:
    return AnswerEntry(
        id=question.id,
        status="answered",
        answer="Decrease.",
        direction="decrease",
        confidence=0.5,
        certainty="low",
        edge_ids=["does_not_exist"],
    )


def test_unverifiable_question_still_gets_an_entry():
    questions = [Question(id="Q1", question="What does S do to O?")]
    answers = run_pipeline(questions, _store(), _stages(_bad_citation_reason))
    assert [a.id for a in answers.answers] == ["Q1"]
    assert answers.answers[0].status == "no_evidence"
    assert answers.answers[0].confidence == 0.0


def test_extras_must_be_existing_relative_paths(tmp_path: Path):
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "ok.html").write_text("x", encoding="utf-8")
    good = AnswerEntry(
        id="Q1", status="no_evidence", answer="none", confidence=0.5,
        searched_for="s", extras=["outputs/ok.html"],
    )
    bad = good.model_copy(update={"extras": ["outputs/missing.html", "/etc/passwd", "../x"]})
    assert check_extras(good, tmp_path) == []
    assert len(check_extras(bad, tmp_path)) == 3
    assert check_entry(good, _store()) == []
