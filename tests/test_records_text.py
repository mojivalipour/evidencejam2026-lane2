"""Retrieval text includes entity types and the edge's own source spans."""

from causa_nostra.graph.models import Edge, Graph, Node
from causa_nostra.graph.store import GraphStore
from causa_nostra.retrieval.records import MAX_SPAN_CHARS, MAX_SPANS, build_record, evidence_span_text


def _store(spans: list[dict]) -> GraphStore:
    return GraphStore(
        Graph(
            graph_id="t",
            nodes=[
                Node(id="S", entity_type="management_intervention", embedding_text="burning"),
                Node(id="O", entity_type="environmental_variable", embedding_text="grass cover"),
            ],
            edges=[
                Edge(
                    id="e1",
                    subject="S",
                    predicate="causes",
                    object="O",
                    causal_language="direct_causal",
                    original_sentence="Burning reduced grass cover.",
                    source_spans=spans,
                )
            ],
        )
    )


def test_record_text_includes_entity_types_and_source_spans() -> None:
    store = _store([{"text": "Fire cut the grass cover sharply."}])
    record = build_record(store.edge("e1"), store)
    assert "subject type: management intervention" in record.text
    assert "object type: environmental variable" in record.text
    assert "source text: Fire cut the grass cover sharply." in record.text


def test_span_text_skips_the_original_sentence_and_duplicates() -> None:
    store = _store([
        {"text": "Burning reduced grass cover."},
        {"text": "Fire cut the grass cover sharply."},
        {"text": "Fire cut the grass cover sharply."},
    ])
    assert evidence_span_text(store.edge("e1")) == "Fire cut the grass cover sharply."


def test_span_text_is_clipped_and_capped() -> None:
    long = "x" * (MAX_SPAN_CHARS + 50)
    spans = [{"text": f"{i} {long}"} for i in range(MAX_SPANS + 2)]
    parts = evidence_span_text(_store(spans).edge("e1")).split(" / ")
    assert len(parts) == MAX_SPANS
    assert all(len(part) <= MAX_SPAN_CHARS for part in parts)


def test_non_dict_spans_are_ignored() -> None:
    store = _store(["plain string", None])  # type: ignore[list-item]
    assert evidence_span_text(store.edge("e1")) == ""


def test_extra_node_terms_are_indexed_once() -> None:
    store = _store([])
    store.node("S").name = "Prescribed burn"
    store.node("S").entity_term = "fire"
    store.node("S").applied_to = [
        {"entity_term": "wd:Q1", "entity_type": "environmental_variable"},
        {"entity_term": "grassland", "entity_type": "environmental_variable"},
    ]
    record = build_record(store.edge("e1"), store, {"wd:Q1": "forest"})
    assert "subject terms: Prescribed burn; fire; forest (wd:Q1); grassland" in record.text
    assert "{" not in record.text
