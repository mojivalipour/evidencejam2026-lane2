"""Indexed, read-only access to a loaded graph.

All retrieval and reasoning code reads the graph through `GraphStore`, so the
raw JSON layout is confined to this package.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path

from causa_nostra.graph.models import Edge, Graph, Node


class GraphStore:
    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        self._nodes: dict[str, Node] = {n.id: n for n in graph.nodes}
        self._edges: dict[str, Edge] = {e.id: e for e in graph.edges}
        self._outgoing: dict[str, list[Edge]] = defaultdict(list)
        self._incoming: dict[str, list[Edge]] = defaultdict(list)
        for edge in graph.edges:
            self._outgoing[edge.subject].append(edge)
            self._incoming[edge.object].append(edge)
        self._document_ids = {
            d["document_id"] for d in graph.source_documents if "document_id" in d
        }

    @classmethod
    def from_json(cls, path: str | Path) -> GraphStore:
        text = Path(path).read_text(encoding="utf-8")
        return cls(Graph.model_validate_json(text))

    # --- lookups -----------------------------------------------------------

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def has_edge(self, edge_id: str) -> bool:
        return edge_id in self._edges

    def has_document(self, document_id: str) -> bool:
        return document_id in self._document_ids

    def node(self, node_id: str) -> Node:
        try:
            return self._nodes[node_id]
        except KeyError:
            raise KeyError(f"unknown node id: {node_id}") from None

    def edge(self, edge_id: str) -> Edge:
        try:
            return self._edges[edge_id]
        except KeyError:
            raise KeyError(f"unknown edge id: {edge_id}") from None

    def edges_from(self, node_id: str) -> list[Edge]:
        return list(self._outgoing.get(node_id, []))

    def edges_to(self, node_id: str) -> list[Edge]:
        return list(self._incoming.get(node_id, []))

    def edges_touching(self, node_id: str) -> list[Edge]:
        return self.edges_from(node_id) + self.edges_to(node_id)

    # --- iteration ---------------------------------------------------------

    def nodes(self) -> Iterable[Node]:
        return self._nodes.values()

    def edges(self) -> Iterable[Edge]:
        return self._edges.values()

    def find_nodes(self, predicate: Callable[[Node], bool]) -> list[Node]:
        return [n for n in self._nodes.values() if predicate(n)]


if __name__ == "__main__":
    # Demo: load the supplied graph, then show one node and its connected edges.
    store = GraphStore.from_json("causal_graph.json")
    first_edge = next(iter(store.edges()))
    print(f"edge {first_edge.id}: {first_edge.subject} -[{first_edge.predicate}]-> {first_edge.object}")
    print(f"edges touching subject: {len(store.edges_touching(first_edge.subject))}")
    print(f"source document known: {store.has_document(first_edge.source_document or '')}")
