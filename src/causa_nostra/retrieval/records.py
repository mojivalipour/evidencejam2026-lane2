"""Readable evidence records: one per edge, carrying everything a reader or LLM needs.

The `text` field is what BM25 indexes. It includes node labels and embedding
text, the predicate and claim strength, null status, context, and the source
sentence, so that search matches on the same content a reviewer would read.
"""

from __future__ import annotations

from pydantic import BaseModel

from causa_nostra.camo_rules.semantics import variable_sign
from causa_nostra.graph.models import Edge
from causa_nostra.graph.store import GraphStore


class EvidenceRecord(BaseModel):
    edge_id: str
    subject_id: str
    object_id: str
    subject_label: str
    object_label: str
    subject_qualifier: str | None
    object_qualifier: str | None
    predicate: str
    causal_language: str
    negated: bool
    relation_sign: str | None  # "positive" | "negative" | None
    source_document: str | None
    original_sentence: str
    ecosystem_context: str | None
    certainty_grade: str | None
    text: str


def _label(node_label: str, fallback: str) -> str:
    return node_label or fallback


def build_record(edge: Edge, store: GraphStore) -> EvidenceRecord:
    subject = store.node(edge.subject)
    obj = store.node(edge.object)
    subject_label = _label(subject.embedding_text, subject.name)
    object_label = _label(obj.embedding_text, obj.name)

    sign = variable_sign(edge, store)
    sign_name = None if sign is None else sign.name.lower()

    null_tag = " NULL RESULT (no relationship found)" if edge.negated else ""
    text = " | ".join(
        part
        for part in (
            f"subject: {subject_label}",
            f"predicate: {edge.predicate} [{edge.causal_language}]{null_tag}",
            f"object: {object_label}",
            f"context: {edge.ecosystem_context}" if edge.ecosystem_context else "",
            f"sentence: {edge.original_sentence}" if edge.original_sentence else "",
        )
        if part
    )

    return EvidenceRecord(
        edge_id=edge.id,
        subject_id=subject.id,
        object_id=obj.id,
        subject_label=subject_label,
        object_label=object_label,
        subject_qualifier=subject.state_or_change_qualifier,
        object_qualifier=obj.state_or_change_qualifier,
        predicate=edge.predicate,
        causal_language=edge.causal_language,
        negated=edge.negated,
        relation_sign=sign_name,
        source_document=edge.source_document,
        original_sentence=edge.original_sentence,
        ecosystem_context=edge.ecosystem_context,
        certainty_grade=edge.certainty_grade,
        text=text,
    )


def build_records(store: GraphStore) -> list[EvidenceRecord]:
    return [build_record(edge, store) for edge in store.edges()]


if __name__ == "__main__":
    # Demo: print the indexed text of the first evidence record.
    store = GraphStore.from_json("causal_graph.json")
    record = build_records(store)[0]
    print(record.model_dump_json(indent=2))
