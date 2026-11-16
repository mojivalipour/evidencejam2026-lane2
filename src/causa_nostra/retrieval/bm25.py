"""BM25 keyword search, backed by Haystack's in-memory BM25 retriever.

Documents and queries pass through the same `normalize`, so "Grasses",
"grass" and "peat_bogs" / "peat bogs" match each other.

Two choices keep results the same across Haystack versions:
- The algorithm is fixed to BM25Okapi. Haystack's default (BM25L) gives every
  document a base score, and Haystack 2.x returns those documents even when
  they share no word with the query.
- A hit is kept only if the document really shares a word with the query. This
  is checked here, not left to Haystack.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from haystack import Document
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

_WORD = re.compile(r"[a-z0-9]+")
_KEEP_S = ("ss", "us", "is")


def _singular(word: str) -> str:
    """Strip a plural ending. Applied to documents and queries alike."""
    if len(word) <= 3:
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith(("sses", "ches", "shes", "xes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith(_KEEP_S):
        return word[:-1]
    return word


def normalize(text: str) -> str:
    """Lowercase, split on anything that is not a letter or digit, and singularise."""
    return " ".join(_singular(word) for word in _WORD.findall(text.lower()))


@dataclass(frozen=True)
class SearchHit:
    doc_id: str
    score: float


class BM25Index:
    """BM25 over `(doc_id, text)` pairs. Only documents sharing a word with the query are returned."""

    def __init__(self, documents: Iterable[tuple[str, str]]) -> None:
        store = InMemoryDocumentStore(bm25_algorithm="BM25Okapi")
        docs = [Document(id=doc_id, content=normalize(text)) for doc_id, text in documents]
        # Haystack rejects empty content, and an empty document can never match.
        docs = [d for d in docs if d.content]
        store.write_documents(docs)
        self._size = len(docs)
        self._words = {d.id: frozenset(d.content.split()) for d in docs}
        self._retriever = InMemoryBM25Retriever(document_store=store)

    def search(self, query: str, top_k: int | None = None) -> list[SearchHit]:
        """Best matches first. `top_k=None` returns every matching document."""
        query = normalize(query)
        if not query or self._size == 0:
            return []
        query_words = set(query.split())
        result = self._retriever.run(query=query, top_k=self._size)
        # A word found in half the documents or more can score 0 in BM25Okapi, so the
        # shared-word test decides what is a hit and the score only decides the order.
        hits = [
            SearchHit(doc_id=d.id, score=max(float(d.score or 0.0), 0.0))
            for d in result["documents"]
            if query_words & self._words[d.id]
        ]
        # Equal scores are ordered by id, so the order never depends on Haystack internals.
        hits.sort(key=lambda hit: (-hit.score, hit.doc_id))
        return hits if top_k is None else hits[:top_k]


if __name__ == "__main__":
    # Demo: broad BM25 search over the supplied graph. Usage: python -m causa_nostra.retrieval.bm25 "query"
    import sys

    from causa_nostra.graph.store import GraphStore
    from causa_nostra.retrieval.records import build_records

    query = " ".join(sys.argv[1:]) or "water table graminoid cover"
    graph = GraphStore.from_json("causal_graph.json")
    index = BM25Index((r.edge_id, r.text) for r in build_records(graph))
    for hit in index.search(query, top_k=5):
        print(f"{hit.score:.3f}  {hit.doc_id}  {graph.edge(hit.doc_id).predicate}")
