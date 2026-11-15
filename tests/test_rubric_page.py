"""The verifier's rubric judgements are written as a page and passed back for the returned answer."""

from pathlib import Path

from causa_nostra.answers.evidence import write_rubric
from causa_nostra.answers.models import AnswerEntry
from causa_nostra.graph.models import Edge, Graph, Node
from causa_nostra.graph.store import GraphStore
from causa_nostra.pipeline.run import answer_question
from causa_nostra.pipeline.stages import Stages
from causa_nostra.pipeline.types import Question, QuestionFrame, Verification


def _entry() -> AnswerEntry:
    return AnswerEntry(id="Q1", status="answered", answer="Decrease.", direction="decrease",
                       confidence=0.5, certainty="low", edge_ids=["e1"])


def test_write_rubric_lists_criteria_problems_and_supporting_edges(tmp_path: Path):
    verification = Verification(
        passed=False,
        criteria={"evidence_match": "partial", "search_coverage": "fail"},
        items={"A1": "pass", "A2": "partial", "D4": "fail"},
        problems=["omits PCBs"],
        supporting_edge_ids=["e1"],
        score=0.55,
    )
    rel = write_rubric(_entry(), verification, tmp_path / "out_evidence", repo_root=tmp_path)

    assert rel == "out_evidence/Q1_rubric.html"
    text = (tmp_path / rel).read_text(encoding="utf-8")
    assert "Does the cited evidence address the intervention" in text  # A1 question text
    assert "Is evidence coverage sufficient" in text  # D4 question text
    assert "partial" in text and "fail" in text
    assert "omits PCBs" in text
    assert "0.550" in text
    assert "e1" in text


def test_rubric_page_lists_all_22_items_for_either_status(tmp_path: Path):
    from causa_nostra.answers.evidence import render_rubric_html

    answered = Verification(passed=True, criteria={}, items={"A1": "pass"}, score=1.0)
    no_evidence_entry = AnswerEntry(id="Q2", status="no_evidence", answer="None.", confidence=0.0,
                                    searched_for="x")
    no_evidence = Verification(passed=True, criteria={}, items={"N1": "pass"}, score=1.0)

    for entry, verification in ((_entry(), answered), (no_evidence_entry, no_evidence)):
        text = render_rubric_html(entry, verification)
        for item in ("A1", "A4", "B1", "C4", "D4", "N1", "N6"):
            assert f"<code>{item}</code>" in text
        assert "not_applicable" in text


def test_on_returned_receives_the_verification_of_the_returned_answer():
    store = GraphStore(
        Graph(
            graph_id="t",
            nodes=[Node(id="S"), Node(id="O")],
            edges=[Edge(id="e1", subject="S", predicate="causes", object="O", causal_language="direct_causal")],
        )
    )

    def reason(question, frame, records, feedback):
        return AnswerEntry(id=question.id, status="answered", answer="Decrease.", direction="decrease",
                           confidence=0.9, certainty="low", edge_ids=["e1"])

    stages = Stages(
        understand=lambda q: QuestionFrame(question_id=q.id, intervention="S", outcome="O"),
        retrieve=lambda frame: [],
        reason=reason,
        verify=lambda q, f, d, r: Verification(passed=True, score=0.8),
    )
    seen: list[tuple[AnswerEntry, Verification]] = []
    entry = answer_question(Question(id="Q1", question="q"), store, stages,
                            on_returned=lambda e, v: seen.append((e, v)))

    assert len(seen) == 1
    assert seen[0][0] == entry
    assert seen[0][1].score == 0.8
