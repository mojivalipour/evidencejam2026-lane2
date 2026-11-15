"""Retrieval as a Haystack pipeline: question frame -> search query -> BM25 documents.

The pipeline is built once per graph and reused for every question. Each run
receives only the frame of one question, so no state is shared between questions.
"""

from __future__ import annotations

from collections.abc import Iterable

from haystack import Document, Pipeline, component
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

from causa_nostra.pipeline.types import QuestionFrame
from causa_nostra.retrieval.records import EvidenceRecord


@component
class FrameToQuery:
    """Turns the intervention, outcome and search terms of a frame into one BM25 query."""

    @component.output_types(query=str)
    def run(self, frame: QuestionFrame) -> dict[str, str]:
        return {"query": search_query(frame)}


def search_query(frame: QuestionFrame) -> str:
    """The exact BM25 query executed for a frame. The verifier logs this same string."""
    parts = [frame.intervention, frame.outcome, *frame.search_terms]
    return " ".join(p for p in parts if p).strip()


class RetrievalPipeline:
    """Haystack BM25 retrieval over evidence records, wired as a two-step pipeline."""

    def __init__(self, records: Iterable[EvidenceRecord], top_k: int = 20) -> None:
        self._records = {r.edge_id: r for r in records}
        self._top_k = top_k

        store = InMemoryDocumentStore()
        store.write_documents(
            [
                Document(id=r.edge_id, content=r.text, meta={"edge_id": r.edge_id})
                for r in self._records.values()
            ]
        )

        self._pipeline = Pipeline()
        self._pipeline.add_component("to_query", FrameToQuery())
        self._pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=store))
        self._pipeline.connect("to_query.query", "retriever.query")

    def retrieve(self, frame: QuestionFrame) -> list[EvidenceRecord]:
        result = self._pipeline.run(
            {
                "to_query": {"frame": frame},
                "retriever": {"top_k": self._top_k},
            }
        )
        return [self._records[doc.id] for doc in result["retriever"]["documents"]]
