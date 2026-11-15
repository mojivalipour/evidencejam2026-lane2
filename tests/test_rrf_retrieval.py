"""Reciprocal rank fusion: both-sided matches outrank one-sided ones, one-sided ones are kept."""

from causa_nostra.pipeline.types import QuestionFrame
from causa_nostra.retrieval.pipeline import RRF_K, RetrievalPipeline
from causa_nostra.retrieval.records import EvidenceRecord


def _record(edge_id: str, subject: str, obj: str, sentence: str = "") -> EvidenceRecord:
    return EvidenceRecord(
        edge_id=edge_id,
        subject_id=f"n-{edge_id}-s",
        object_id=f"n-{edge_id}-o",
        subject_label=subject,
        object_label=obj,
        subject_qualifier=None,
        object_qualifier=None,
        predicate="causes",
        causal_language="causal",
        negated=False,
        relation_sign="positive",
        source_document=None,
        original_sentence=sentence,
        ecosystem_context=None,
        certainty_grade=None,
        text=f"subject: {subject} | object: {obj} | sentence: {sentence}",
    )


# Fixture graph: one edge matching both sides, one matching only the intervention,
# one matching only the outcome, and one matching neither.
BOTH = _record("a-both", "prescribed burning fire", "grass cover", "burning raised grass cover")
SUBJECT_ONLY = _record("b-subject", "prescribed burning fire", "tree density", "burning changed tree density")
OBJECT_ONLY = _record("c-object", "grazing cattle", "grass cover", "grazing changed grass cover")
UNRELATED = _record("d-other", "road salt", "frog count", "salt reduced frogs")
RECORDS = [BOTH, SUBJECT_ONLY, OBJECT_ONLY, UNRELATED]


def _frame(intervention: str = "burning", outcome: str = "grass cover") -> QuestionFrame:
    return QuestionFrame(question_id="T-1", intervention=intervention, outcome=outcome)


def test_both_sided_match_ranks_first():
    hits = RetrievalPipeline(RECORDS).retrieve(_frame())
    assert hits[0].edge_id == "a-both"


def test_one_sided_matches_are_kept_but_ranked_lower():
    ids = [r.edge_id for r in RetrievalPipeline(RECORDS).retrieve(_frame())]
    assert "b-subject" in ids
    assert "c-object" in ids
    assert ids.index("a-both") < ids.index("b-subject")
    assert ids.index("a-both") < ids.index("c-object")


def test_unrelated_edge_is_not_returned():
    ids = [r.edge_id for r in RetrievalPipeline(RECORDS).retrieve(_frame())]
    assert "d-other" not in ids


def test_rank_trace_records_each_signal_that_found_the_edge():
    hits = {r.edge_id: r for r in RetrievalPipeline(RECORDS).retrieve(_frame())}
    assert set(hits["a-both"].retrieval_ranks) == {"subject", "object", "fulltext"}
    assert "object" not in hits["b-subject"].retrieval_ranks
    assert "subject" not in hits["c-object"].retrieval_ranks


def test_input_records_are_not_mutated():
    pipeline = RetrievalPipeline(RECORDS)
    pipeline.retrieve(_frame())
    assert BOTH.retrieval_ranks == {}


def test_top_k_limits_the_output():
    hits = RetrievalPipeline(RECORDS, top_k=2).retrieve(_frame())
    assert len(hits) == 2
    assert hits[0].edge_id == "a-both"


def test_empty_intervention_still_searches_by_outcome():
    hits = RetrievalPipeline(RECORDS).retrieve(_frame(intervention="", outcome="grass cover"))
    ids = [r.edge_id for r in hits]
    assert ids[:2] == ["a-both", "c-object"]


def test_ties_are_ordered_by_edge_id_and_stable():
    twin_a = _record("t-1", "wildfire", "moss")
    twin_b = _record("t-2", "wildfire", "moss")
    pipeline = RetrievalPipeline([twin_b, twin_a])
    frame = _frame(intervention="wildfire", outcome="moss")
    first = [r.edge_id for r in pipeline.retrieve(frame)]
    assert first == ["t-1", "t-2"]
    assert first == [r.edge_id for r in pipeline.retrieve(frame)]


def test_fused_score_formula_for_top_edge():
    # The top edge is rank 1 in subject, object and full text, so its fused score is 3 / (RRF_K + 1).
    # This documents the formula; the test checks the ranks, not the float value.
    hits = RetrievalPipeline(RECORDS).retrieve(_frame())
    assert hits[0].retrieval_ranks == {"subject": 1, "object": 1, "fulltext": 1}
    assert RRF_K == 60
