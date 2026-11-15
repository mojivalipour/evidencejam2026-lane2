"""Agent tests with a scripted fake LLM client. No network calls."""

import json

import pytest

from causa_nostra.agents import AgentOutputError, QuestionProcessor, Reasoner, Verifier, parse_json_object, rubric_score
from causa_nostra.answers.models import AnswerEntry
from causa_nostra.answers.rubric import ANSWERED_QUESTIONS, item_texts
from causa_nostra.llm.client import ChatResult
from causa_nostra.pipeline.types import Question, QuestionFrame
from causa_nostra.retrieval.records import EvidenceRecord


class FakeClient:
    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.calls: list[list[dict]] = []

    @staticmethod
    def _with_reasons(text: str) -> str:
        """Verifier replies: give every returned item a reason, so tests state only judgements."""
        data = json.loads(text) if text.lstrip().startswith("{") else None
        if isinstance(data, dict) and isinstance(data.get("items"), dict) and "reasons" not in data:
            data["reasons"] = {item: "reason" for item in data["items"]}
            return json.dumps(data)
        return text

    def chat(self, messages, *, temperature=0.0, max_tokens=None, thinking=True):
        self.calls.append(messages)
        return ChatResult(text=self._with_reasons(self.replies.pop(0)))


def _record(edge_id: str = "e1", negated: bool = False) -> EvidenceRecord:
    return EvidenceRecord(
        edge_id=edge_id, subject_id="S", object_id="O", subject_label="burning", object_label="grass cover",
        subject_qualifier=None, object_qualifier=None, predicate="reduces", causal_language="direct_causal",
        negated=negated, relation_sign="negative", source_document="d1", original_sentence="Burning reduced grass.",
        ecosystem_context=None, certainty_grade=None, text=f"subject: burning | predicate: reduces | object: grass cover [{edge_id}]",
        retrieval_ranks={"subject": 1, "object": 1},  # matched both sides by default
    )


QUESTION = Question(id="H-1", question="What does burning do to non-native grass cover?")
FRAME = QuestionFrame(question_id="H-1", intervention="burning", outcome="grass cover")


# --- base parsing -----------------------------------------------------------

def test_parse_accepts_code_fence_and_surrounding_prose():
    assert parse_json_object('Sure:\n```json\n{"a": 1}\n```') == {"a": 1}


def test_parse_rejects_non_object_and_garbage():
    with pytest.raises(AgentOutputError):
        parse_json_object("[1, 2]")
    with pytest.raises(AgentOutputError):
        parse_json_object("no json here")


# --- question processor -----------------------------------------------------

def test_question_processor_builds_frame_and_caps_search_terms():
    reply = json.dumps({
        "intervention": "burning", "outcome": "non-native grass cover", "context": [],
        "search_terms": [f"term{i}" for i in range(30)] + ["Burning", "burning"],
    })
    frame = QuestionProcessor(FakeClient(reply)).run(QUESTION)
    assert frame.question_id == "H-1"
    assert frame.intervention == "burning"
    assert len(frame.search_terms) == 15


def test_question_processor_requires_intervention_and_outcome():
    reply = json.dumps({"intervention": "burning", "outcome": ""})
    with pytest.raises(AgentOutputError):
        QuestionProcessor(FakeClient(reply)).run(QUESTION)


# --- reasoner ---------------------------------------------------------------

def test_reasoner_returns_validated_entry_and_passes_feedback():
    reply = json.dumps({
        "status": "answered", "answer": "Burning lowers grass cover.", "direction": "decrease",
        "confidence": 0.7, "certainty": "low", "edge_ids": ["e1"], "searched_for": None,
    })
    client = FakeClient(reply)
    frame = QuestionFrame(question_id="H-1", intervention="burning", outcome="grass cover")
    entry = Reasoner(client).run(QUESTION, frame, [_record()], feedback=["cite fewer edges"])
    assert isinstance(entry, AnswerEntry)
    assert entry.id == "H-1" and entry.edge_ids == ["e1"]
    user_prompt = client.calls[0][1]["content"]
    assert "[e1]" in user_prompt
    assert "cite fewer edges" in user_prompt


def test_reasoner_rejects_inconsistent_answered_entry():
    reply = json.dumps({"status": "answered", "answer": "x", "direction": None, "confidence": 0.5,
                        "certainty": None, "edge_ids": []})
    frame = QuestionFrame(question_id="H-1", intervention="a", outcome="b")
    with pytest.raises(ValueError):
        Reasoner(FakeClient(reply)).run(QUESTION, frame, [_record()], feedback=[])


# --- verifier ---------------------------------------------------------------

def _draft() -> AnswerEntry:
    return AnswerEntry(id="H-1", status="answered", answer="Lower.", direction="decrease",
                       confidence=0.7, certainty="low", edge_ids=["e1"])


def _items(**overrides: str) -> dict[str, str]:
    base = {item: "pass" for category in ANSWERED_QUESTIONS.values() for item in category}
    base |= {"N1": "irrelevant", "N2": "irrelevant", "N3": "irrelevant", "N4": "irrelevant",
             "N5": "irrelevant", "N6": "irrelevant"}
    return base | overrides


def test_verifier_passes_when_no_gated_item_fails():
    reply = json.dumps({
        "items": _items(D4="partial"),
        "problems": [], "supporting_edge_ids": ["e1", "made_up"],
    })
    result = Verifier(FakeClient(reply)).run(QUESTION, FRAME, _draft(), [_record()])
    assert result.passed is True
    assert result.supporting_edge_ids == ["e1"]


def test_verifier_fails_on_wrong_direction():
    reply = json.dumps({
        "items": _items(B1="fail"),
        "problems": ["draft says decrease, record is increase"],
    })
    result = Verifier(FakeClient(reply)).run(QUESTION, FRAME, _draft(), [_record()])
    assert result.passed is False
    assert result.problems == ["draft says decrease, record is increase"]


def test_verifier_missing_items_count_as_fail():
    # Both the first reply and the retry omit every item: all items fail.
    empty = json.dumps({"items": {}, "problems": []})
    result = Verifier(FakeClient(empty, empty)).run(QUESTION, FRAME, _draft(), [_record()])
    assert result.passed is False
    assert result.criteria["citation_support"] == "fail"
    assert result.problems[0] == "verifier left items unanswered: " + ", ".join(item_texts(True))


def test_verifier_gate_failure_without_problems_gets_fallback_message():
    reply = json.dumps({"items": _items(C1="fail"), "problems": []})
    result = Verifier(FakeClient(reply)).run(QUESTION, FRAME, _draft(), [_record()])
    assert result.passed is False
    assert result.problems == ["gated items failed: C1"]


def test_verifier_drops_non_string_and_unknown_supporting_ids():
    reply = json.dumps({"items": _items(), "problems": [], "supporting_edge_ids": ["e1", 3, None, "ghost"]})
    result = Verifier(FakeClient(reply)).run(QUESTION, FRAME, _draft(), [_record()])
    assert result.supporting_edge_ids == ["e1"]


def test_verifier_non_dict_items_and_non_list_problems_are_handled():
    reply = json.dumps({"items": ["pass"], "problems": "oops"})
    result = Verifier(FakeClient(reply, reply)).run(QUESTION, FRAME, _draft(), [_record()])
    assert result.passed is False
    assert result.problems[0].startswith("verifier left items unanswered")



def test_rubric_score_is_weighted_in_python():
    all_pass = {"evidence_match": "pass", "direction_and_causal_language": "pass",
                "citation_support": "pass", "search_coverage": "pass"}
    assert rubric_score(all_pass) == pytest.approx(1.0)
    assert rubric_score({**all_pass, "direction_and_causal_language": "partial"}) == pytest.approx(0.85)
    assert rubric_score({**all_pass, "search_coverage": "fail"}) == pytest.approx(0.85)


def test_verifier_returns_rubric_score():
    result = Verifier(FakeClient(json.dumps({"items": _items(D4="partial"), "problems": []}))).run(
        QUESTION, FRAME, _draft(), [_record()])
    assert result.score == pytest.approx(0.925)


def test_verifier_intervention_mismatch_cannot_pass_as_irrelevant():
    reply = json.dumps({"items": _items(A1="irrelevant"), "problems": []})
    result = Verifier(FakeClient(reply)).run(QUESTION, FRAME, _draft(), [_record()])
    assert result.passed is False
    assert result.criteria["evidence_match"] == "fail"


def _no_evidence_draft() -> AnswerEntry:
    return AnswerEntry(id="H-1", status="no_evidence", answer="None found.", confidence=0.5,
                       searched_for="burning, grass cover, fire")


_A_IRRELEVANT = {item: "irrelevant" for category in ANSWERED_QUESTIONS.values() for item in category}
_N_PASS = _A_IRRELEVANT | {"N1": "pass", "N2": "pass", "N3": "pass", "N4": "pass", "N5": "pass", "N6": "pass"}


def test_verifier_no_evidence_draft_uses_missing_evidence_prompt_and_rubric():
    client = FakeClient(json.dumps({"items": _N_PASS, "problems": []}))
    result = Verifier(client).run(QUESTION, FRAME, _no_evidence_draft(), [_record()])
    assert result.passed is True
    assert result.score == pytest.approx(1.0)
    assert set(result.criteria) == {"interpretation", "search_coverage", "candidates_checked"}
    system = client.calls[0][0]["content"]
    assert '"N6"' in system
    assert '"A1"' not in system  # only the draft's own items are asked: keeps verification fast
    assert "A1" not in result.items  # other rubric's items are not judged for this draft


def test_verifier_no_evidence_fails_when_evidence_was_found():
    # N5 = fail means the search found evidence that answers the question.
    reply = json.dumps({"items": _N_PASS | {"N5": "fail"}, "problems": ["record e1 answers the question"]})
    result = Verifier(FakeClient(reply)).run(QUESTION, FRAME, _no_evidence_draft(), [_record()])
    assert result.passed is False
    assert result.problems == ["record e1 answers the question"]


def test_verifier_no_evidence_missing_items_fail():
    reply = json.dumps({"items": {"N6": "pass"}, "problems": []})
    result = Verifier(FakeClient(reply, reply)).run(QUESTION, FRAME, _no_evidence_draft(), [_record()])
    assert result.passed is False
    assert result.score == pytest.approx(0.0)


def test_verifier_retries_once_for_unanswered_items_and_uses_the_complete_reply():
    partial = json.dumps({"items": {"A1": "pass"}, "problems": []})
    complete = json.dumps({"items": _items(), "problems": []})
    client = FakeClient(partial, complete)
    result = Verifier(client).run(QUESTION, FRAME, _draft(), [_record()])
    assert len(client.calls) == 2
    assert "left these ids" in client.calls[1][1]["content"]
    assert result.passed is True
    assert set(result.item_reasons) == {item for category in ANSWERED_QUESTIONS.values() for item in category}


def test_verifier_item_without_reason_counts_as_fail():
    no_reason = json.dumps({"items": _items(), "reasons": {"A1": "shown in e1"}, "problems": []})
    client = FakeClient(no_reason, no_reason)
    result = Verifier(client).run(QUESTION, FRAME, _draft(), [_record()])
    assert result.passed is False
    assert result.items["A2"] == "fail"
    assert result.item_reasons["A1"] == "shown in e1"
    assert result.item_reasons["A2"] == "no reason returned"


def test_verifier_no_evidence_cannot_pass_when_it_names_supporting_edges():
    # Regression: all items passed, yet the verifier listed an edge that matched both sides.
    reply = json.dumps({"items": _N_PASS, "problems": [], "supporting_edge_ids": ["e1"]})
    both = _record().model_copy(update={"retrieval_ranks": {"subject": 1, "object": 2}})
    result = Verifier(FakeClient(reply)).run(QUESTION, FRAME, _no_evidence_draft(), [both])
    assert result.passed is False
    assert result.supporting_edge_ids == ["e1"]
    assert "bear on the question" in result.problems[0]


def test_verifier_no_evidence_ignores_one_sided_candidates():
    # An edge that matched only one side (e.g. the right outcome, another intervention) is not supporting evidence.
    reply = json.dumps({"items": _N_PASS, "problems": [], "supporting_edge_ids": ["e1"]})
    one_sided = _record().model_copy(update={"retrieval_ranks": {"object": 1}})
    result = Verifier(FakeClient(reply)).run(QUESTION, FRAME, _no_evidence_draft(), [one_sided])
    assert result.passed is True
    assert result.supporting_edge_ids == []


def test_verifier_receives_structured_metadata_and_search_log():
    client = FakeClient(json.dumps({"items": _items(), "problems": []}))
    frame = QuestionFrame(question_id="H-1", intervention="burning", outcome="grass cover",
                          search_terms=["fire", "wildfire"])
    Verifier(client).run(QUESTION, frame, _draft(), [_record()])
    user = client.calls[0][1]["content"]
    assert "relation_sign: negative" in user
    assert "source_document: d1" in user
    assert "subject_qualifier: (none)" in user
    assert "subject side: [burning fire wildfire] | object side: [grass cover fire wildfire]" in user
    assert "Alternative terms searched: fire, wildfire" in user
    assert "Retrieved edge_ids (top k, ranked): e1" in user


def test_pipeline_accepts_verified_draft_with_rubric_confidence():
    from causa_nostra.graph.models import Graph, Node
    from causa_nostra.graph.store import GraphStore
    from causa_nostra.pipeline.run import answer_question
    from causa_nostra.pipeline.stages import Stages
    from causa_nostra.pipeline.types import Question, Verification

    store = GraphStore(Graph(graph_id="t", nodes=[Node(id="S"), Node(id="O")], edges=[]))
    stages = Stages(
        understand=lambda q: QuestionFrame(question_id=q.id, intervention="S", outcome="O"),
        retrieve=lambda frame: [],
        reason=lambda q, f, r, fb: _no_evidence_draft().model_copy(update={"id": q.id}),
        verify=lambda q, f, d, r: Verification(passed=True, score=0.85),
    )
    entry = answer_question(Question(id="H-1", question="q"), store, stages)
    assert entry.confidence == pytest.approx(0.85)
