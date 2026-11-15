"""Inline citations and per-question evidence pages. No network calls."""

from pathlib import Path

from causa_nostra.agents.reasoner import with_citations
from causa_nostra.answers.evidence import render_evidence_html, write_evidence
from causa_nostra.answers.models import AnswerEntry
from causa_nostra.graph.models import Edge, Graph, Node
from causa_nostra.graph.store import GraphStore
from causa_nostra.retrieval.records import build_record


def _store() -> GraphStore:
    return GraphStore(
        Graph(
            graph_id="t",
            nodes=[Node(id="S", label="burning"), Node(id="O", label="grass cover")],
            edges=[Edge(id="e1", subject="S", predicate="causes", object="O", causal_language="direct_causal")],
        )
    )


def _answered(edge_ids: list[str], answer: str = "Burning reduced grass cover.") -> AnswerEntry:
    return AnswerEntry(
        id="Q1", status="answered", answer=answer, direction="decrease",
        confidence=0.5, certainty="low", edge_ids=edge_ids,
    )


def test_with_citations_keeps_answer_when_every_edge_is_named():
    assert with_citations("Burning reduced cover [e1].", ["e1"]) == "Burning reduced cover [e1]."


def test_with_citations_appends_missing_edges():
    assert with_citations("Burning reduced cover.", ["e1", "e2"]) == "Burning reduced cover. [e1; e2]"


def test_with_citations_no_edges_leaves_answer_unchanged():
    assert with_citations("No evidence.", []) == "No evidence."


def test_with_citations_matches_whole_tokens_only():
    # "e1" is inside "e10" in the answer, so it must still be appended.
    assert with_citations("Reduced cover [e10].", ["e1", "e10"]) == "Reduced cover [e10]. [e1]"


def test_colliding_question_ids_get_distinct_pages(tmp_path: Path):
    store = _store()
    first = write_evidence(_answered(["e1"]).model_copy(update={"id": "Q/1"}), store, tmp_path, repo_root=tmp_path)
    second = write_evidence(_answered(["e1"]).model_copy(update={"id": "Q:1"}), store, tmp_path, repo_root=tmp_path)
    assert first != second
    assert len(list(tmp_path.iterdir())) == 2


def test_negated_edge_is_marked_as_null_result(tmp_path: Path):
    store = _store()
    record = build_record(store.edge("e1"), store).model_copy(update={"negated": True})
    page = render_evidence_html(_answered(["e1"]), [record])
    assert "NULL RESULT: causes" in page


def test_write_evidence_creates_page_with_source_sentence(tmp_path: Path):
    store = _store()
    path = write_evidence(_answered(["e1"]), store, tmp_path / "evidence", repo_root=tmp_path)
    assert path is not None and path.endswith("Q1_evidence.html")
    assert (tmp_path / path).exists()


def test_write_evidence_skips_no_evidence_entries(tmp_path: Path):
    entry = AnswerEntry(id="Q2", status="no_evidence", answer="None.", confidence=0.0, searched_for="s")
    assert write_evidence(entry, _store(), tmp_path, repo_root=tmp_path) is None
    assert not list(tmp_path.iterdir())


def test_write_evidence_outside_repo_returns_none(tmp_path: Path):
    other = tmp_path / "elsewhere"
    other.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    assert write_evidence(_answered(["e1"]), _store(), other, repo_root=repo) is None


def test_render_escapes_html_in_answer(tmp_path: Path):
    store = _store()
    entry = _answered(["e1"], answer="<script>x</script>")
    page = render_evidence_html(entry, [])
    assert "<script>x</script>" not in page
    assert "&lt;script&gt;" in page
