"""Deterministic graph invariants from the CAMO schema.

These checks report problems in the supplied graph and in any edge we cite.
They never modify the graph. Errors in the graph are not penalised, but they
must be flagged in the evidence trace.
"""

from __future__ import annotations

from dataclasses import dataclass

from causa_nostra.graph.models import Edge
from causa_nostra.graph.store import GraphStore

NON_CAUSAL_PREDICATES = frozenset({"associated_with", "correlated_with", "precedes"})
NON_CAUSAL_LANGUAGE = frozenset({"associational", "no_relationship"})


@dataclass(frozen=True)
class Issue:
    edge_id: str
    rule: str
    message: str


def check_edge(edge: Edge, store: GraphStore) -> list[Issue]:
    issues: list[Issue] = []

    def add(rule: str, message: str) -> None:
        issues.append(Issue(edge.id, rule, message))

    for role, node_id in (("subject", edge.subject), ("object", edge.object)):
        if not store.has_node(node_id):
            add("dangling_node", f"{role} {node_id} is not a node in the graph")

    if edge.source_document and not store.has_document(edge.source_document):
        add("unknown_source_document", f"{edge.source_document} is not in source_documents")

    # Canonical null-result encoding: negated <=> no_relationship.
    if edge.negated != (edge.causal_language == "no_relationship"):
        add("null_flag_mismatch", "negated must be true iff causal_language is no_relationship")

    if edge.predicate in NON_CAUSAL_PREDICATES and edge.causal_language not in NON_CAUSAL_LANGUAGE:
        add("non_causal_predicate_strength", f"{edge.predicate} cannot carry {edge.causal_language}")

    if edge.predicate == "causes" and edge.causal_language == "associational":
        add("causes_with_associational", "causes cannot carry associational language")

    if edge.negated and store.has_node(edge.object):
        if store.node(edge.object).state_or_change_qualifier == "unchanged":
            add("null_encoded_as_unchanged", "null results must not use the unchanged qualifier")

    return issues


def check_graph(store: GraphStore) -> list[Issue]:
    issues: list[Issue] = []
    for edge in store.edges():
        issues.extend(check_edge(edge, store))
    return issues


if __name__ == "__main__":
    # Run all graph invariants on the supplied graph and print a summary.
    from collections import Counter

    store = GraphStore.from_json("causal_graph.json")
    issues = check_graph(store)
    print(f"{len(issues)} issue(s)")
    for rule, count in Counter(i.rule for i in issues).most_common():
        print(f"  {rule}: {count}")
