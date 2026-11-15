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

from causa_nostra.camo_rules.semantics import NEG, POS, UNC, Polarity, predicate_polarity, variable_sign
from causa_nostra.graph.models import Edge
from causa_nostra.graph.store import GraphStore

_CURIE = r"(?:wd|ENVO|PATO|CHEBI|GO|NCBITaxon):[A-Za-z0-9_]+"
_LABELLED_CURIE = re.compile(rf"([^|:()]+?) \(({_CURIE})\)")
_BARE_CURIE = re.compile(rf"(?<![(\w])({_CURIE})(?![\w)])")
# Edge source spans added to the indexed text: at most this many, each clipped to this many characters.
MAX_SPANS = 3
MAX_SPAN_CHARS = 300

# The object node's qualifier is the state the edge reports for the outcome. That
# state is not the effect on its own: "increased X prevents increased Y" reports
# Y as increased, but the predicate inhibits that increase. The effect is the
# predicate sign times the reported outcome state (schema: FCM sign composition).
_OUTCOME_STATE: dict[str, Polarity] = {"increased": POS, "decreased": NEG, "unchanged": UNC}
# A subject with one of these qualifiers is a reduction or removal of the driver,
# so the outcome direction describes what happens when the driver goes DOWN.
_REDUCTION_QUALIFIERS = frozenset({"decreased", "absent", "removed", "terminated", "interrupted", "aborted"})
# Wording in the source sentence that points the other way from the predicate.
_INHIBIT_WORDS = re.compile(r"\b(?:prevent|inhibit|suppress|block|hinder|curb)")
_PROMOTE_WORDS = re.compile(r"\b(?:caus|promot|enhanc|induc|stimulat|drive)")


def outcome_direction(negated: bool, predicate: str, object_qualifier: str | None) -> str:
    """Direction of the outcome effect this edge reports.

    "no_effect" for a null result. "stable" when the outcome is reported as
    unchanged. Otherwise the predicate sign times the reported outcome state:
    "increase" or "decrease". "unclear" when the outcome has no reported state
    or the predicate has no sign (associations, regulation, mediation).
    """
    if negated:
        return "no_effect"
    state = _OUTCOME_STATE.get(object_qualifier or "")
    if state is None:
        return "unclear"
    if state is UNC:
        return "stable"
    sign = predicate_polarity(predicate)
    if sign is None:
        return "unclear"
    return "increase" if sign.value * state.value > 0 else "decrease"


def direction_ambiguity(predicate: str, sentence: str) -> str | None:
    """Flag a source sentence whose wording contradicts the predicate sign.

    This is a check only. A conflict is reported for review, not resolved.
    """
    sign = predicate_polarity(predicate)
    text = sentence.lower()
    if sign is POS and _INHIBIT_WORDS.search(text):
        return "sentence uses inhibiting wording but the predicate is positive"
    if sign is NEG and _PROMOTE_WORDS.search(text):
        return "sentence uses promoting wording but the predicate is negative"
    return None


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
    direction_ambiguity: str | None = None  # set when the sentence contradicts the predicate sign
    subject_is_reduction: bool = False
    source_document: str | None
    original_sentence: str
    ecosystem_context: str | None
    certainty_grade: str | None
    text: str
    # Set per question by retrieval: the 1-based rank of this edge in each signal
    # (e.g. {"subject": 3, "object": 12, "fulltext": 1}). Empty for unretrieved records.
    retrieval_ranks: dict[str, int] = {}


def _readable(value: str | None) -> str:
    """Enum-style code as words: 'management_intervention' -> 'management intervention'."""
    return (value or "").replace("_", " ").strip()


def evidence_span_text(edge: Edge) -> str:
    """The edge's own source-span text, whitespace-normalised, clipped, deduplicated and joined for indexing.

    The original sentence is skipped here because it is already indexed on its own.
    """
    seen: list[str] = []
    original = " ".join((edge.original_sentence or "").split())
    for span in edge.source_spans:
        raw = span.get("text") if isinstance(span, dict) else None
        text = " ".join(str(raw or "").split())[:MAX_SPAN_CHARS]
        if text and text != original and text not in seen:
            seen.append(text)
        if len(seen) == MAX_SPANS:
            break
    return " / ".join(seen)


def _extra_terms(node, labels: dict[str, str], covered: str) -> str:
    """Node name, entity term and applied-to values not already in its embedding text, joined for indexing."""
    raw: list[str] = [node.name or "", node.entity_term or ""]
    applied = node.applied_to if isinstance(node.applied_to, list) else []
    for item in applied:
        if isinstance(item, dict):
            raw.append(str(item.get("entity_term") or ""))
        elif item:
            raw.append(str(item))
    terms: list[str] = []
    for value in raw:
        term = label_curies(" ".join(value.split()), labels)
        if term and term.lower() not in covered.lower() and term not in terms:
            terms.append(term)
    return "; ".join(terms)


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
    direction = outcome_direction(edge.negated, edge.predicate, obj.state_or_change_qualifier)
    ambiguity = None if edge.negated else direction_ambiguity(edge.predicate, edge.original_sentence or "")
    if ambiguity and direction in ("increase", "decrease"):
        direction = "unclear"  # flagged for review, not labelled from the qualifier alone
    reduction = subject.state_or_change_qualifier in _REDUCTION_QUALIFIERS

    null_tag = " NULL RESULT (no relationship found)" if edge.negated else ""
    context = (edge.ecosystem_context or "").replace("_", " ")
    subject_type = _readable(subject.entity_type)
    object_type = _readable(obj.entity_type)
    spans = evidence_span_text(edge)
    subject_extra = _extra_terms(subject, labels, subject_label)
    object_extra = _extra_terms(obj, labels, object_label)
    text = " | ".join(
        part
        for part in (
            f"subject: {subject_label}",
            f"subject terms: {subject_extra}" if subject_extra else "",
            f"subject type: {subject_type}" if subject_type else "",
            "note: subject is a reduction or removal of the driver" if reduction else "",
            f"predicate: {edge.predicate} [{edge.causal_language}]{null_tag}",
            f"object: {object_label}",
            f"object terms: {object_extra}" if object_extra else "",
            f"object type: {object_type}" if object_type else "",
            f"reported outcome state: {obj.state_or_change_qualifier}" if obj.state_or_change_qualifier else "",
            f"outcome direction: {direction}",
            f"ambiguity: {ambiguity}" if ambiguity else "",
            f"context: {context}" if context else "",
            f"sentence: {edge.original_sentence}" if edge.original_sentence else "",
            f"source text: {spans}" if spans else "",
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
        direction_ambiguity=ambiguity,
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
        "outcome_direction": record.outcome_direction,
        "direction_ambiguity": record.direction_ambiguity,
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
