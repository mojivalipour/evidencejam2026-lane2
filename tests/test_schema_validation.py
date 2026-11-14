"""Tests that the supplied graph and deliberately invalid graphs are checked against the CAMO LinkML schema."""

from __future__ import annotations

import json
from pathlib import Path

from causa_nostra.camo_rules.schema import validate_graph_file

REPO_ROOT = Path(__file__).resolve().parents[1]
SUPPLIED_GRAPH = REPO_ROOT / "causal_graph.json"


def test_supplied_graph_validates_against_camo_schema() -> None:
    assert validate_graph_file(SUPPLIED_GRAPH) == []


def test_graph_missing_required_identifier_fails(tmp_path: Path) -> None:
    graph = json.loads(SUPPLIED_GRAPH.read_text(encoding="utf-8"))
    del graph["graph_id"]  # graph_id is required and is the identifier on CausalGraph
    broken = tmp_path / "missing_graph_id.json"
    broken.write_text(json.dumps(graph), encoding="utf-8")

    problems = validate_graph_file(broken)

    assert problems, "a graph without graph_id must fail LinkML validation"
