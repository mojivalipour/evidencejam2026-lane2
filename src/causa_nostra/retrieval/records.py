"""Readable evidence records: one per edge, carrying everything a reader or LLM needs.

`text` is what the reasoner reads and what the broad BM25 search indexes. It
holds the node labels, the predicate and claim strength, the reported outcome
direction, the context and the source sentence.

Ontology codes that a node gives without a label (e.g. `applied to: wd:Q123`)
are given the label the graph uses for the same code elsewhere, so that keyword
search can match them.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from pydantic import BaseModel

from causa_nostra.camo_rules.semantics import variable_sign
from causa_nostra.graph.models import Edge
from causa_nostra.graph.store import GraphStore

_CURIE = r"(?:wd|ENVO|PATO|CHEBI|GO|NCBITaxon):[A-Za-z0-9_]+"
_LABELLED_CURIE = re.compile(rf"([^|:()]+?) \(({_CURIE})\)")
_BARE_CURIE = re.compile(rf"(?<![(\w])({_CURIE})(?![\w)])")

# Outcome direction as this graph records it: the object node's qualifier holds
# the reported outcome, whatever the predicate (also for prevents / disrupts and
# for associated_with / correlated_with). Qualifiers not listed are "unclear".
_OUTCOME_DIRECTION = {"increased": "increase", "decreased": "decrease", "unchanged": "stable"}
# A subject with one of these qualifiers is a reduction or removal of the driver,
# so the outcome direction describes what happens when the driver goes DOWN.
_REDUCTION_QUALIFIERS = frozenset({"decreased", "absent", "removed", "terminated", "interrupted", "aborted"})


def outcome_direction(negated: bool, object_qualifier: str | None) -> str:
    """What an edge reports about its outcome node.

    "no_effect" for a null result, otherwise "increase", "decrease", "stable"
    (reported as unchanged) or "unclear" (no directional qualifier).
    """
    if negated:
        return "no_effect"
    return _OUTCOME_DIRECTION.get(object_qualifier, "unclear")


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
    relation_sign: str | None  # "positive" | "negative" | None (schema sign arithmetic)
    outcome_direction: str = "unclear"  # "increase" | "decrease" | "no_effect" | "stable" | "unclear"
    subject_is_reduction: bool = False
    source_document: str | None
    original_sentence: str
    ecosystem_context: str | None
    certainty_grade: str | None
    text: str


def curie_labels(store: GraphStore) -> dict[str, str]:
    """Map each ontology code to the label the graph most often writes beside it."""
    seen: dict[str, Counter[str]] = defaultdict(Counter)
    for node in store.nodes():
        for label, curie in _LABELLED_CURIE.findall(node.embedding_text):
            seen[curie][label.strip()] += 1
    return {curie: counts.most_common(1)[0][0] for curie, counts in seen.items()}


def label_curies(text: str, labels: dict[str, str]) -> str:
    """Rewrite bare codes as `label (code)` where the label is known."""

    def expand(match: re.Match[str]) -> str:
        curie = match.group(1)
        return f"{labels[curie]} ({curie})" if curie in labels else curie

    return _BARE_CURIE.sub(expand, text)


def build_record(edge: Edge, store: GraphStore, labels: dict[str, str] | None = None) -> EvidenceRecord:
    labels = labels or {}
    subject = store.node(edge.subject)
    obj = store.node(edge.object)
    subject_label = label_curies(subject.embedding_text or subject.name, labels)
    object_label = label_curies(obj.embedding_text or obj.name, labels)

    sign = variable_sign(edge, store)
    direction = outcome_direction(edge.negated, obj.state_or_change_qualifier)
    reduction = subject.state_or_change_qualifier in _REDUCTION_QUALIFIERS

    null_tag = " NULL RESULT (no relationship found)" if edge.negated else ""
    context = (edge.ecosystem_context or "").replace("_", " ")
    text = " | ".join(
        part
        for part in (
            f"subject: {subject_label}",
            "note: subject is a reduction or removal of the driver" if reduction else "",
            f"predicate: {edge.predicate} [{edge.causal_language}]{null_tag}",
            f"object: {object_label}",
            f"outcome direction: {direction}",
            f"context: {context}" if context else "",
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
        relation_sign=None if sign is None else sign.name.lower(),
        outcome_direction=direction,
        subject_is_reduction=reduction,
        source_document=edge.source_document,
        original_sentence=edge.original_sentence,
        ecosystem_context=edge.ecosystem_context,
        certainty_grade=edge.certainty_grade,
        text=text,
    )


def build_records(store: GraphStore) -> list[EvidenceRecord]:
    labels = curie_labels(store)
    return [build_record(edge, store, labels) for edge in store.edges()]


def structured_text(record: EvidenceRecord) -> str:
    """Every authoritative field of a record, one per line, for LLM judgement.

    Unlike `text` (the BM25 indexing string), this keeps qualifiers, relation sign,
    certainty grade and source document, so direction and provenance can be checked.
    """
    fields = {
        "edge_id": record.edge_id,
        "subject": record.subject_label,
        "subject_qualifier": record.subject_qualifier,
        "predicate": record.predicate,
        "causal_language": record.causal_language,
        "null_result": record.negated,
        "relation_sign": record.relation_sign,
        "object": record.object_label,
        "object_qualifier": record.object_qualifier,
        "ecosystem_context": record.ecosystem_context,
        "certainty_grade": record.certainty_grade,
        "source_document": record.source_document,
        "original_sentence": record.original_sentence,
    }
    return "\n".join(f"  {name}: {'(none)' if value in (None, '') else value}" for name, value in fields.items())


if __name__ == "__main__":
    # Demo: print the first evidence record.
    store = GraphStore.from_json("causal_graph.json")
    record = build_records(store)[0]
    print(record.model_dump_json(indent=2))
