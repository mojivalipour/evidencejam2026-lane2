"""CAMO direction and null-result interpretation, using explicit graph cases."""

import pytest

from causa_nostra.camo_rules.checks import check_edge
from causa_nostra.camo_rules.semantics import (
    NEG,
    POS,
    UnknownVocabularyError,
    is_causal_claim,
    variable_sign,
)
from causa_nostra.graph.models import Edge, Graph, Node
from causa_nostra.graph.store import GraphStore
from causa_nostra.retrieval.records import build_record, direction_ambiguity, outcome_direction


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


@pytest.mark.parametrize(
    ("pred", "obj_q", "expected"),
    [
        ("prevents", "increased", "decrease"),  # increased X prevents increased Y: Y is held down
        ("prevents", "decreased", "increase"),  # prevents a decrease: Y is held up
        ("causes", "decreased", "decrease"),
        ("causes", "increased", "increase"),
        ("causes", "unchanged", "stable"),
        ("associated_with", "increased", "unclear"),  # unsigned predicate: no composed direction
        ("causes", None, "unclear"),  # no reported outcome state
    ],
)
def test_outcome_direction_combines_predicate_and_outcome_state(pred, obj_q, expected):
    assert outcome_direction(False, pred, obj_q) == expected


def test_outcome_direction_null_result_is_no_effect():
    assert outcome_direction(True, "causes", "increased") == "no_effect"


def test_record_reverses_increased_prevents_increased():
    edge = _edge("e", "S", "prevents", "O")
    store = _store({"S": "increased", "O": "increased"}, [edge])
    assert build_record(edge, store).outcome_direction == "decrease"


@pytest.mark.parametrize(
    ("pred", "sentence", "flagged"),
    [
        ("causes", "Burning prevents grass growth.", True),
        ("prevents", "Burning promotes grass growth.", True),
        ("causes", "Burning increased grass cover.", False),
        ("prevents", "Increased burning prevented an increase in grass cover.", False),
        ("causes", "", False),
    ],
)
def test_direction_ambiguity_flags_sentence_predicate_conflict(pred, sentence, flagged):
    assert (direction_ambiguity(pred, sentence) is not None) is flagged


def test_record_marks_conflicting_sentence_unclear():
    edge = _edge("e", "S", "causes", "O")
    edge = edge.model_copy(update={"original_sentence": "Burning prevents grass growth."})
    store = _store({"S": None, "O": "increased"}, [edge])
    record = build_record(edge, store)
    assert record.outcome_direction == "unclear"
    assert record.direction_ambiguity is not None
