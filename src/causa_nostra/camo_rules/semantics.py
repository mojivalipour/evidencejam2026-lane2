"""CAMO interpretation rules: polarity of predicates, qualifiers and edges.

Source of truth: `schema/causalmosaic.yaml` (CausalPredicateEnum,
StateOrChangeQualifierEnum). The variable-level composition used here is:

    edge sign = subject polarity x predicate polarity x object polarity

Rules that matter for correctness:
- A node with no qualifier, `occurred` or `ongoing` counts as POSITIVE
  (the node is the variable's reference state).
- `unchanged` on a node means the state was stable, not a tested null result.
  It has no variable-level sign.
- Null results (`negated`) produce no sign at all.
- Predicate alone never determines outcome direction.

Unknown vocabulary raises instead of guessing, so schema drift is visible.
"""

from __future__ import annotations

from enum import Enum

from causa_nostra.graph.models import Edge, Node
from causa_nostra.graph.store import GraphStore


class UnknownVocabularyError(ValueError):
    """A predicate or qualifier is not in the CAMO vocabulary this module knows."""


class Polarity(Enum):
    POSITIVE = 1
    NEGATIVE = -1
    UNCHANGED = 0  # stable state; carries no variable-level sign


POS, NEG, UNC = Polarity.POSITIVE, Polarity.NEGATIVE, Polarity.UNCHANGED

# Predicate polarity. None = unsigned (association, regulation, mediation, ...).
PREDICATE_POLARITY: dict[str, Polarity | None] = {
    "causes": POS,
    "contributes_to": POS,
    "enables": POS,
    "prevents": NEG,
    "disrupts": NEG,
    "regulates": None,  # sign depends on the node states
    "mediates": None,  # pathway, not a sign
    "moderates": None,  # modifier, not a sign
    "associated_with": None,
    "correlated_with": None,
    "precedes": None,
}

# Qualifier polarity. The None key is a node with no qualifier.
QUALIFIER_POLARITY: dict[str | None, Polarity] = {
    None: POS,
    "increased": POS,
    "present": POS,
    "introduced": POS,
    "initiated": POS,
    "occurred": POS,
    "ongoing": POS,
    "decreased": NEG,
    "absent": NEG,
    "removed": NEG,
    "terminated": NEG,
    "interrupted": NEG,
    "aborted": NEG,
    "unchanged": UNC,
}

CAUSAL_LANGUAGE = frozenset({"direct_causal", "uncertain_causal"})


def predicate_polarity(predicate: str) -> Polarity | None:
    try:
        return PREDICATE_POLARITY[predicate]
    except KeyError:
        raise UnknownVocabularyError(f"unknown predicate: {predicate!r}") from None


def node_polarity(node: Node) -> Polarity:
    qualifier = node.state_or_change_qualifier
    try:
        return QUALIFIER_POLARITY[qualifier]
    except KeyError:
        raise UnknownVocabularyError(
            f"unknown state/change qualifier on {node.id}: {qualifier!r}"
        ) from None


def is_causal_claim(edge: Edge) -> bool:
    """True for a positive, non-null claim phrased causally."""
    return not edge.negated and edge.causal_language in CAUSAL_LANGUAGE


def variable_sign(edge: Edge, store: GraphStore) -> Polarity | None:
    """Variable-level sign of an edge, or None when the edge carries no sign.

    None is returned for null results, unsigned predicates, and any edge with
    an unchanged node on either end.
    """
    if edge.negated:
        return None
    predicate_sign = predicate_polarity(edge.predicate)
    if predicate_sign is None:
        return None
    subject_sign = node_polarity(store.node(edge.subject))
    object_sign = node_polarity(store.node(edge.object))
    if UNC in (subject_sign, object_sign):
        return None
    return Polarity(predicate_sign.value * subject_sign.value * object_sign.value)


if __name__ == "__main__":
    # Demo: distribution of variable-level signs across the supplied graph.
    from collections import Counter

    store = GraphStore.from_json("causal_graph.json")
    counts = Counter(
        (variable_sign(e, store).name if variable_sign(e, store) else "none")
        for e in store.edges()
    )
    for name, count in counts.most_common():
        print(f"{name}: {count}")
