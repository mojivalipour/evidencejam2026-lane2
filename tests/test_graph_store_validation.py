from __future__ import annotations

import json
import logging

import pytest
from pydantic import ValidationError

from causa_nostra.graph.store import GraphStore


def _graph_data() -> dict:
    return {
        "graph_id": "test",
        "nodes": [
            {
                "id": "source",
                "entity_term": "source entity",
                "measured_attribute": "abundance",
                "applied_to": [],
                "state_or_change_qualifier": "increased",
            },
            {
                "id": "target",
                "entity_term": "target entity",
                "measured_attribute": "cover",
                "applied_to": [],
                "state_or_change_qualifier": "decreased",
            },
        ],
        "edges": [
            {
                "id": "edge-1",
                "subject": "source",
                "object": "target",
                "predicate": "causes",
                "negated": False,
                "causal_language": "direct_causal",
                "comparator": [],
                "original_sentence": "The source caused the target to change.",
            }
        ],
    }


def test_missing_essential_fields_are_logged_and_graph_still_loads(
    tmp_path, caplog
) -> None:
    data = _graph_data()
    del data["nodes"][0]["measured_attribute"]
    del data["edges"][0]["comparator"]
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        store = GraphStore.from_json(path)

    assert store.has_node("source")
    assert any(
        "measured_attribute" in message and "source" in message
        for message in caplog.messages
    )
    assert any(
        "comparator" in message and "edge-1" in message
        for message in caplog.messages
    )


def test_dangling_edge_node_reference_is_logged_and_graph_still_loads(
    tmp_path, caplog
) -> None:
    data = _graph_data()
    data["edges"][0]["object"] = "missing-node"
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        store = GraphStore.from_json(path)

    assert store.has_edge("edge-1")
    assert any(
        "object references" in message and "missing-node" in message
        for message in caplog.messages
    )


def test_missing_graph_file_is_logged_and_raises(tmp_path, caplog) -> None:
    path = tmp_path / "missing.json"

    with caplog.at_level(logging.ERROR), pytest.raises(FileNotFoundError):
        GraphStore.from_json(path)

    assert any("Graph file not found" in message for message in caplog.messages)


def test_invalid_graph_model_is_logged_and_raises(tmp_path, caplog) -> None:
    path = tmp_path / "invalid.json"
    path.write_text('{"graph_id":', encoding="utf-8")

    with caplog.at_level(logging.ERROR), pytest.raises(ValidationError):
        GraphStore.from_json(path)

    assert any("Invalid graph data" in message for message in caplog.messages)
