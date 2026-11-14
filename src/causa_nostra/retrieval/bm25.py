"""BM25 search over evidence records, backed by Haystack.

Haystack is imported lazily so that graph checks and tests run without the
`retrieval` extra installed.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from haystack import Document
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

from causa_nostra.retrieval.records import EvidenceRecord


@dataclass(frozen=True)
class SearchHit:
    edge_id: str
    score: float


class BM25Index:
    """Haystack in-memory BM25 over `EvidenceRecord.text`."""

    def __init__(self, records: Iterable[EvidenceRecord]) -> None:
        self._store = InMemoryDocumentStore()
        self._store.write_documents(
            [
                Document(id=r.edge_id, content=r.text, meta={"edge_id": r.edge_id})
                for r in records
            ]
        )
        self._retriever = InMemoryBM25Retriever(document_store=self._store)

    def search(self, query: str, top_k: int = 20) -> list[SearchHit]:
        result = self._retriever.run(query=query, top_k=top_k)
        return [SearchHit(edge_id=d.id, score=float(d.score or 0.0)) for d in result["documents"]]


if __name__ == "__main__":
    # Demo: BM25 search over the supplied graph. Usage: python -m causa_nostra.retrieval.bm25 "query"
    import sys

    from causa_nostra.graph.store import GraphStore
    from causa_nostra.retrieval.records import build_records

    query = " ".join(sys.argv[1:]) or "water table graminoid cover"
    store = GraphStore.from_json("causal_graph.json")
    index = BM25Index(build_records(store))
    for hit in index.search(query, top_k=5):
        print(f"{hit.score:.3f}  {hit.edge_id}  {store.edge(hit.edge_id).predicate}")
