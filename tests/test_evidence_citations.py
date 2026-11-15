"""Inline citations and per-question evidence pages. No network calls."""

from pathlib import Path

from causa_nostra.agents.reasoner import with_citations
from causa_nostra.answers.evidence import render_evidence_html, write_evidence
from causa_nostra.answers.models import AnswerEntry
from causa_nostra.graph.models import Edge, Graph, Node
from causa_nostra.graph.store import GraphStore


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
