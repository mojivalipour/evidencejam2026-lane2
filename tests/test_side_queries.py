"""Synonyms stay on the side they describe, so a reversed edge cannot match both sides."""

from causa_nostra.pipeline.types import QuestionFrame
from causa_nostra.retrieval.pipeline import side_queries


def test_side_terms_stay_on_their_side():
    frame = QuestionFrame(
        question_id="H-1",
        intervention="burning",
        outcome="grass cover",
        intervention_terms=["fire"],
        outcome_terms=["herbaceous cover"],
    )
    subject, obj = side_queries(frame)
    assert subject == "burning fire"
    assert obj == "grass cover herbaceous cover"


def test_legacy_mixed_terms_are_used_on_both_sides():
    frame = QuestionFrame(question_id="H-1", intervention="burning", outcome="grass cover", search_terms=["fire"])
    subject, obj = side_queries(frame)
    assert subject == "burning fire"
    assert obj == "grass cover fire"
