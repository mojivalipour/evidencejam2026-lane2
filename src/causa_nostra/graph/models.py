"""Pydantic models for `causal_graph.json`.

Field names follow the exported JSON. Fields whose inner structure is not
needed by the pipeline are typed loosely so that exporter drift does not break
loading; the CAMO layer validates vocabulary values explicitly.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _Record(BaseModel):
    # Keep unknown keys so that the raw record is never silently lost.
    model_config = ConfigDict(extra="allow")


class Node(_Record):
    """A CAMO node: entity + measured attribute + state/change qualifier."""

    id: str
    name: str = ""
    entity_term: str | None = None
    entity_type: str | None = None
    measured_attribute: str | None = None
    state_or_change_qualifier: str | None = None
    embedding_text: str = ""
    applied_to: Any = None
    source_spans: list[Any] = Field(default_factory=list)
    annotator: str | None = None


class Edge(_Record):
    """A causal claim from one source document, between two nodes."""

    id: str
    subject: str
    predicate: str
    object: str
    negated: bool = False
    causal_language: str
    source_document: str | None = None
    original_sentence: str = ""
    annotation_notes: str | None = None
    philosophical_accounts: list[str] = Field(default_factory=list)
    source_spans: list[Any] = Field(default_factory=list)
    evidential_basis: dict[str, Any] = Field(default_factory=dict)
    comparator: list[Any] = Field(default_factory=list)
    strength: dict[str, Any] | None = None
    ecosystem_context: str | None = None
    certainty_grade: str | None = None
    annotator: str | None = None


class Graph(_Record):
    """The full corpus-level graph."""

    graph_id: str
    schema_version: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
    source_documents: list[dict[str, Any]] = Field(default_factory=list)


if __name__ == "__main__":
    # Smoke test: parse the supplied graph and print its size.
    from pathlib import Path

    graph = Graph.model_validate_json(Path("causal_graph.json").read_text(encoding="utf-8"))
    print(f"graph {graph.graph_id}: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
