"""Retrieval: question frame -> evidence records, ranked by reciprocal rank fusion (RRF).

Three BM25 signals score every record for one frame:
- subject: the intervention words against the subject labels,
- object: the outcome words against the object labels,
- fulltext: all question words against the full record text.

Each signal gives every matching edge a 1-based rank. An edge's fused score is
the sum over signals of 1 / (RRF_K + rank), so an edge ranked well by several
signals (e.g. matching both sides) rises to the top. An edge matching only one
side is still returned, ranked lower. Ranks are used, not raw BM25 scores, so no
threshold needs tuning.

Each returned record carries `retrieval_ranks`, the per-signal ranks, so the
search log shows which signal found it. There is one code path for every frame;
a frame with an empty side simply yields no hits from that side's signal.

The indexes are built once per graph. Each call sees one frame only.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from causa_nostra.pipeline.types import QuestionFrame
from causa_nostra.retrieval.bm25 import BM25Index
from causa_nostra.retrieval.records import EvidenceRecord

# Standard RRF constant: dampens the gap between top ranks (Cormack et al., 2009).
RRF_K = 60


def side_queries(frame: QuestionFrame) -> tuple[str, str]:
    """The subject-side and object-side queries for a frame.

    Each side uses only its own synonyms, so an outcome synonym cannot match a
    subject node. A frame with neither list (legacy frames carrying only the
    mixed `search_terms`) uses those terms on both sides.
    """
    intervention_terms = list(frame.intervention_terms)
    outcome_terms = list(frame.outcome_terms)
    if not intervention_terms and not outcome_terms:
        intervention_terms = outcome_terms = list(frame.search_terms)
    subject_query = " ".join([frame.intervention, *intervention_terms]).strip()
    object_query = " ".join([frame.outcome, *outcome_terms]).strip()
    return subject_query, object_query


def search_query(frame: QuestionFrame) -> str:
    """Human-readable record of the side queries, for the verifier's log."""
    subject_query, object_query = side_queries(frame)
    return f"subject side: [{subject_query}] | object side: [{object_query}]"


class RetrievalPipeline:
    """Reciprocal-rank-fusion retrieval over subject, object and full-text BM25 signals."""

    def __init__(self, records: Iterable[EvidenceRecord], top_k: int = 20) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self._records = {r.edge_id: r for r in records}
        self._top_k = top_k

        self._subjects = BM25Index((r.edge_id, r.subject_label) for r in self._records.values())
        self._objects = BM25Index((r.edge_id, r.object_label) for r in self._records.values())
        self._full_text = BM25Index((r.edge_id, r.text) for r in self._records.values())

    def retrieve(self, frame: QuestionFrame) -> list[EvidenceRecord]:
        subject_query, object_query = side_queries(frame)
        fulltext_query = " ".join(
            [frame.intervention, frame.outcome, *frame.intervention_terms, *frame.outcome_terms, *frame.search_terms]
        )
        signals = {
            "subject": self._subjects.search(subject_query),
            "object": self._objects.search(object_query),
            "fulltext": self._full_text.search(fulltext_query),
        }

        fused: dict[str, float] = defaultdict(float)
        ranks: dict[str, dict[str, int]] = defaultdict(dict)
        for name, hits in signals.items():
            for rank, hit in enumerate(hits, start=1):
                fused[hit.doc_id] += 1.0 / (RRF_K + rank)
                ranks[hit.doc_id][name] = rank

        # Highest fused score first; edge id breaks ties so the order is stable.
        ordered = sorted(fused, key=lambda edge_id: (-fused[edge_id], edge_id))[: self._top_k]
        return [self._records[edge_id].model_copy(update={"retrieval_ranks": ranks[edge_id]}) for edge_id in ordered]


if __name__ == "__main__":
    # Demo: python -m causa_nostra.retrieval.pipeline "herbicide" "native plant cover"
    import sys

    from causa_nostra.graph.store import GraphStore
    from causa_nostra.retrieval.records import build_records

    intervention, outcome = sys.argv[1:3] if len(sys.argv) >= 3 else ("herbicide", "native plant cover")
    pipeline = RetrievalPipeline(build_records(GraphStore.from_json("causal_graph.json")))
    frame = QuestionFrame(question_id="demo", intervention=intervention, outcome=outcome)
    for record in pipeline.retrieve(frame):
        print(f"{record.edge_id}  {record.retrieval_ranks}  {record.subject_label[:50]} -> {record.object_label[:60]}")
