"""Fault-tolerance tests: a failing question must not drop the batch, and output stays valid."""

import json
from pathlib import Path

from causa_nostra.answers.models import AnswerEntry, AnswersFile
from causa_nostra.graph.models import Edge, Graph, Node
from causa_nostra.graph.store import GraphStore
from causa_nostra.pipeline.batch import run_batch, sanitize_entry, write_answers_atomic
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


def _answered(question, frame, records, feedback, edges=("e1",), extras=()) -> AnswerEntry:
    return AnswerEntry(
        id=question.id,
        status="answered",
        answer="Decrease.",
        direction="decrease",
        confidence=0.5,
        certainty="low",
        edge_ids=list(edges),
        extras=list(extras),
    )


QUESTIONS = [Question(id="Q1", question="a"), Question(id="Q2", question="b"), Question(id="Q3", question="c")]


def test_crash_in_one_question_does_not_drop_others(tmp_path: Path):
    def reason(question, frame, records, feedback):
        if question.id == "Q2":
            raise RuntimeError("endpoint timed out")
        return _answered(question, frame, records, feedback)

    out = tmp_path / "answers.json"
    answers = run_batch(QUESTIONS, _store(), _stages(reason), out, repo_root=tmp_path)

    assert [a.id for a in answers.answers] == ["Q1", "Q2", "Q3"]
    by_id = {a.id: a for a in answers.answers}
    assert by_id["Q1"].status == "answered"
    assert by_id["Q2"].status == "no_evidence"
    assert "RuntimeError" in by_id["Q2"].answer
    assert by_id["Q3"].status == "answered"


def test_final_file_on_disk_is_complete_and_parseable(tmp_path: Path):
    def reason(question, frame, records, feedback):
        raise ValueError("bad json")

    out = tmp_path / "answers.json"
    run_batch(QUESTIONS, _store(), _stages(reason), out, repo_root=tmp_path)

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["team"] == "CAUSA NOSTRA"
    assert [a["id"] for a in data["answers"]] == ["Q1", "Q2", "Q3"]
    assert not list(tmp_path.glob("answers.json.*.tmp"))


def test_unknown_citation_is_dropped_and_valid_one_kept(tmp_path: Path):
    def reason(question, frame, records, feedback):
        return _answered(question, frame, records, feedback, edges=("e1", "made_up"))

    # Verification in answer_question rejects the bad draft first, so the revision
    # must repair it. Here we test sanitize_entry directly on a draft with a bad citation.
    draft = _answered(QUESTIONS[0], None, [], [], edges=("e1", "made_up"))
    cleaned, warnings = sanitize_entry(draft, _store(), tmp_path)
    assert cleaned.edge_ids == ["e1"]
    assert cleaned.status == "answered"
    assert any("made_up" in w for w in warnings)


def test_answered_with_only_invalid_citations_becomes_fallback(tmp_path: Path):
    draft = _answered(QUESTIONS[0], None, [], [], edges=("made_up",))
    cleaned, _ = sanitize_entry(draft, _store(), tmp_path)
    assert cleaned.status == "no_evidence"
    assert cleaned.edge_ids == []
    assert cleaned.searched_for


def test_invalid_extra_is_dropped(tmp_path: Path):
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "ok.html").write_text("x", encoding="utf-8")
    draft = _answered(QUESTIONS[0], None, [], [], extras=("outputs/ok.html", "outputs/missing.html", "/etc/passwd"))
    cleaned, warnings = sanitize_entry(draft, _store(), tmp_path)
    assert cleaned.extras == ["outputs/ok.html"]
    assert len(warnings) == 2


def test_atomic_write_leaves_no_temp_files_and_replaces_content(tmp_path: Path):
    out = tmp_path / "answers.json"
    out.write_text("old", encoding="utf-8")
    write_answers_atomic(out, AnswersFile(answers=[]))
    assert json.loads(out.read_text(encoding="utf-8"))["answers"] == []
    assert not list(tmp_path.glob("answers.json.*.tmp"))
