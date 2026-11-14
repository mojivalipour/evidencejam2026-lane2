"""Retrieval: question frame -> evidence records that match BOTH sides of the question.

A question asks what an intervention does to an outcome. An edge is a candidate
only if its subject node matches the intervention AND its object node matches
the outcome. Searching the two sides separately keeps out edges that match one
side only, or that match the wrong way round (the outcome as the cause).

Steps for one frame:
1. Search subject texts with the intervention words, object texts with the outcome words.
2. Keep edges found on both sides whose score on each side is at least
   `min_relative_score` of the best score on that side.
3. If nothing is left, broaden once: keep every edge found on both sides.
4. Rank by the sum of the two relative scores and return the best `top_k`.

An empty list means no edge links the two sides. A frame with an empty
intervention or an empty outcome (the no-LLM placeholder) cannot be searched by
side, so it falls back to one broad search over the full record text.

The index is built once per graph. Each call sees one frame only.
"""

from __future__ import annotations

from collections.abc import Iterable

from causa_nostra.pipeline.types import QuestionFrame
from causa_nostra.retrieval.bm25 import BM25Index
from causa_nostra.retrieval.records import EvidenceRecord


def side_queries(frame: QuestionFrame) -> tuple[str, str]:
    """The subject-side and object-side queries for a frame.

    `QuestionFrame.search_terms` is one mixed list, so its terms are added to
    both sides. If a frame carries `intervention_terms` / `outcome_terms`
    (not part of `QuestionFrame` today), those are used instead, which keeps an
    outcome synonym from matching a subject node.
    """
    intervention_terms = list(getattr(frame, "intervention_terms", None) or [])
    outcome_terms = list(getattr(frame, "outcome_terms", None) or [])
    if not intervention_terms and not outcome_terms:
        intervention_terms = outcome_terms = list(frame.search_terms)
    subject_query = " ".join([frame.intervention, *intervention_terms]).strip()
    object_query = " ".join([frame.outcome, *outcome_terms]).strip()
    return subject_query, object_query


def _relative_scores(index: BM25Index, query: str) -> dict[str, float]:
    """Score of each matching document as a share of the best score (0 to 1)."""
    hits = index.search(query)
    if not hits:
        return {}
    best = hits[0].score
    if best <= 0:  # every match scored 0 (only very common words matched): treat them as equal
        return {hit.doc_id: 1.0 for hit in hits}
    return {hit.doc_id: hit.score / best for hit in hits}


class RetrievalPipeline:
    """Two-sided BM25 retrieval over evidence records."""

    def __init__(
        self,
        records: Iterable[EvidenceRecord],
        top_k: int = 20,
        min_relative_score: float = 0.3,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not 0.0 <= min_relative_score <= 1.0:
            raise ValueError("min_relative_score must be between 0 and 1")
        self._records = {r.edge_id: r for r in records}
        self._top_k = top_k
        self._min_relative_score = min_relative_score

        self._subjects = BM25Index((r.edge_id, r.subject_label) for r in self._records.values())
        self._objects = BM25Index((r.edge_id, r.object_label) for r in self._records.values())
        self._full_text = BM25Index((r.edge_id, r.text) for r in self._records.values())

    def retrieve(self, frame: QuestionFrame) -> list[EvidenceRecord]:
        if not frame.intervention.strip() or not frame.outcome.strip():
            return self._broad_search(frame)

        subject_query, object_query = side_queries(frame)
        subject_scores = _relative_scores(self._subjects, subject_query)
        object_scores = _relative_scores(self._objects, object_query)
        both = [edge_id for edge_id in subject_scores if edge_id in object_scores]

        strong = [
            edge_id
            for edge_id in both
            if subject_scores[edge_id] >= self._min_relative_score
            and object_scores[edge_id] >= self._min_relative_score
        ]
        chosen = strong or both  # broaden once when the strict filter leaves nothing

        # Highest combined score first; edge id breaks ties so the order is stable.
        chosen.sort(key=lambda edge_id: (-(subject_scores[edge_id] + object_scores[edge_id]), edge_id))
        return [self._records[edge_id] for edge_id in chosen[: self._top_k]]

    def _broad_search(self, frame: QuestionFrame) -> list[EvidenceRecord]:
        """One query over the full record text, for frames without two sides."""
        query = " ".join([frame.intervention, frame.outcome, *frame.search_terms])
        hits = self._full_text.search(query, top_k=self._top_k)
        return [self._records[hit.doc_id] for hit in hits]


if __name__ == "__main__":
    # Demo: python -m causa_nostra.retrieval.pipeline "herbicide" "native plant cover"
    import sys

    from causa_nostra.graph.store import GraphStore
    from causa_nostra.retrieval.records import build_records

    intervention, outcome = sys.argv[1:3] if len(sys.argv) >= 3 else ("herbicide", "native plant cover")
    pipeline = RetrievalPipeline(build_records(GraphStore.from_json("causal_graph.json")))
    frame = QuestionFrame(question_id="demo", intervention=intervention, outcome=outcome)
    for record in pipeline.retrieve(frame):
        print(f"{record.edge_id}  [{record.outcome_direction}]  {record.subject_label[:50]} -> {record.object_label[:60]}")
