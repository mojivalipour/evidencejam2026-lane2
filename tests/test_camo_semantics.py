"""CAMO direction and null-result interpretation, using explicit graph cases."""

import pytest

from causa_nostra.camo.checks import check_edge
from causa_nostra.camo.semantics import (
    NEG,
    POS,
    UnknownVocabularyError,
    is_causal_claim,
    variable_sign,
)
from causa_nostra.graph.models import Edge, Graph, Node
from causa_nostra.graph.store import GraphStore


def _store(nodes: dict[str, str | None], edges: list[Edge]) -> GraphStore:
    graph = Graph(
        graph_id="test",
        nodes=[Node(id=k, name=k, state_or_change_qualifier=v) for k, v in nodes.items()],
        edges=edges,
    )
    return GraphStore(graph)


def _edge(eid: str, subj: str, pred: str, obj: str, language: str = "direct_causal", negated: bool = False) -> Edge:
    return Edge(id=eid, subject=subj, predicate=pred, object=obj, causal_language=language, negated=negated)


@pytest.mark.parametrize(
    ("subj_q", "pred", "obj_q", "expected"),
    [
        ("decreased", "causes", "decreased", POS),  # decrease causes decrease: same-sign
        ("increased", "prevents", "increased", NEG),  # prevents reverses the sign
        ("decreased", "causes", None, NEG),  # no qualifier on object counts as +1
        ("removed", "disrupts", "increased", POS),  # two negatives and a positive
        ("increased", "prevents", "unchanged", None),  # stable state has no sign
        ("increased", "regulates", "increased", None),  # regulation is not a sign
        ("increased", "associated_with", "increased", None),  # association is not causal
    ],
)
def test_variable_sign_follows_camo_composition(subj_q, pred, obj_q, expected):
    store = _store({"S": subj_q, "O": obj_q}, [_edge("e", "S", pred, "O")])
    assert variable_sign(store.edge("e"), store) is expected


def test_null_result_has_no_sign_and_is_not_causal():
    edge = _edge("e", "S", "causes", "O", language="no_relationship", negated=True)
    store = _store({"S": "increased", "O": "increased"}, [edge])
    assert variable_sign(edge, store) is None
    assert not is_causal_claim(edge)


def test_association_is_not_a_causal_claim():
    edge = _edge("e", "S", "associated_with", "O", language="associational")
    assert not is_causal_claim(edge)


def test_unknown_qualifier_raises_instead_of_guessing():
    store = _store({"S": "elevated", "O": None}, [_edge("e", "S", "causes", "O")])
    with pytest.raises(UnknownVocabularyError):
        variable_sign(store.edge("e"), store)


def test_checks_flag_causes_with_associational_language():
    edge = _edge("e", "S", "causes", "O", language="associational")
    store = _store({"S": None, "O": None}, [edge])
    rules = {issue.rule for issue in check_edge(edge, store)}
    assert "causes_with_associational" in rules


def test_checks_flag_null_encoded_as_unchanged():
    edge = _edge("e", "S", "causes", "O", language="no_relationship", negated=True)
    store = _store({"S": "increased", "O": "unchanged"}, [edge])
    rules = {issue.rule for issue in check_edge(edge, store)}
    assert "null_encoded_as_unchanged" in rules
