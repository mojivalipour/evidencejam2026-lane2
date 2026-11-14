"""Indexed, read-only access to a loaded graph.

All retrieval and reasoning code reads the graph through `GraphStore`, so the
raw JSON layout is confined to this package.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path

from pydantic import ValidationError

from causa_nostra.graph.models import Edge, Graph, Node

log = logging.getLogger(__name__)

_REQUIRED_NODE_FIELDS = (
    "entity_term",
    "measured_attribute",
    "applied_to",
    "state_or_change_qualifier",
)
_REQUIRED_EDGE_FIELDS = (
    "subject",
    "object",
    "predicate",
    "negated",
    "causal_language",
    "comparator",
    "original_sentence",
)


def _is_missing_required_field(record: Node | Edge, field: str) -> bool:
    if field not in record.model_fields_set:
        return True
    value = getattr(record, field)
    return value is None or (isinstance(value, str) and not value.strip())


def _validation_warnings(graph: Graph) -> tuple[str, ...]:
    warnings: list[str] = []
    for kind, records, fields in (
        ("node", graph.nodes, _REQUIRED_NODE_FIELDS),
        ("edge", graph.edges, _REQUIRED_EDGE_FIELDS),
    ):
        for field in fields:
            missing_ids = [
                record.id
                for record in records
                if _is_missing_required_field(record, field)
            ]
            if missing_ids:
                examples = ", ".join(missing_ids[:5])
                more = " ..." if len(missing_ids) > 5 else ""
                warnings.append(
                    f"{len(missing_ids)} {kind}(s) missing required field "
                    f"{field!r}; IDs: {examples}{more}"
                )

    node_ids = {node.id for node in graph.nodes}
    for role in ("subject", "object"):
        dangling = [
            f"{edge.id}->{getattr(edge, role)}"
            for edge in graph.edges
            if getattr(edge, role) not in node_ids
        ]
        if dangling:
            examples = ", ".join(dangling[:5])
            more = " ..." if len(dangling) > 5 else ""
            warnings.append(
                f"{len(dangling)} edge(s) have {role} references to unknown "
                f"node IDs; edge->node: {examples}{more}"
            )
    return tuple(warnings)


class GraphStore:
    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        self.validation_warnings = _validation_warnings(graph)
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
        graph_path = Path(path)
        try:
            text = graph_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            log.exception("Graph file not found: %s", graph_path)
            raise
        except PermissionError:
            log.exception("Permission denied reading graph file: %s", graph_path)
            raise
        except UnicodeDecodeError:
            log.exception("Graph file is not valid UTF-8: %s", graph_path)
            raise
        except OSError:
            log.exception("Could not read graph file: %s", graph_path)
            raise

        try:
            graph = Graph.model_validate_json(text)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(part) for part in error['loc']) or '<graph>'}: "
                f"{error['msg']}"
                for error in exc.errors(include_input=False)
            )
            log.error("Invalid graph data in %s: %s", graph_path, problems)
            raise

        store = cls(graph)
        for warning in store.validation_warnings:
            log.warning("Graph validation warning for %s: %s", graph_path, warning)
        return store

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
