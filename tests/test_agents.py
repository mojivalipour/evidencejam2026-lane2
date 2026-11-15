"""Agent tests with a scripted fake LLM client. No network calls."""

import json

import pytest

from causa_nostra.agents import AgentOutputError, QuestionProcessor, Reasoner, Verifier, parse_json_object, rubric_score
from causa_nostra.answers.models import AnswerEntry
from causa_nostra.answers.rubric import ANSWERED_QUESTIONS
from causa_nostra.llm.client import ChatResult
from causa_nostra.pipeline.types import Question, QuestionFrame
from causa_nostra.retrieval.records import EvidenceRecord


class FakeClient:
    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.calls: list[list[dict]] = []

    def chat(self, messages, *, temperature=0.0, max_tokens=None, thinking=True):
        self.calls.append(messages)
        return ChatResult(text=self.replies.pop(0))


def _record(edge_id: str = "e1", negated: bool = False) -> EvidenceRecord:
    return EvidenceRecord(
        edge_id=edge_id, subject_id="S", object_id="O", subject_label="burning", object_label="grass cover",
        subject_qualifier=None, object_qualifier=None, predicate="reduces", causal_language="direct_causal",
        negated=negated, relation_sign="negative", source_document="d1", original_sentence="Burning reduced grass.",
        ecosystem_context=None, certainty_grade=None, text=f"subject: burning | predicate: reduces | object: grass cover [{edge_id}]",
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
    result = Verifier(FakeClient(json.dumps({"items": {}, "problems": []}))).run(QUESTION, FRAME, _draft(), [_record()])
    assert result.passed is False
    assert result.criteria["citation_support"] == "fail"



def test_rubric_score_is_weighted_in_python():
    all_pass = {"evidence_match": "pass", "direction_and_causal_language": "pass",
                "citation_support": "pass", "search_coverage": "pass"}
    assert rubric_score(all_pass) == pytest.approx(1.0)
    assert rubric_score({**all_pass, "direction_and_causal_language": "partial"}) == pytest.approx(0.85)
    assert rubric_score({**all_pass, "search_coverage": "fail"}) == pytest.approx(0.85)
