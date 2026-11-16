"""LinkML validation of graph data against `schema/causalmosaic.yaml`.

The CAMO schema is the source of truth for graph structure, enums and
constraints. This module runs the LinkML validator on a graph file and returns
the problems as strings, so the CLI and pipeline can report them in the
evidence trace.
"""

from __future__ import annotations

from pathlib import Path

from linkml.validator import validate_file

DEFAULT_SCHEMA = Path(__file__).resolve().parents[3] / "schema" / "causalmosaic.yaml"
ROOT_CLASS = "CausalGraph"


def validate_graph_file(
    graph_path: str | Path,
    schema_path: str | Path = DEFAULT_SCHEMA,
) -> list[str]:
    """Return LinkML validation messages for a graph file. Empty list means valid."""
    report = validate_file(
        file=str(graph_path),
        schema=str(schema_path),
        target_class=ROOT_CLASS,
    )
    return [str(result.message) for result in report.results]


if __name__ == "__main__":
    # Demo: validate the supplied graph against the CAMO schema.
    problems = validate_graph_file("causal_graph.json")
    print(f"{len(problems)} LinkML problem(s)")
    for message in problems[:20]:
        print(f"  {message}")
