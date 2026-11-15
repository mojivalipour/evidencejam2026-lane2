"""LinkML validation of graph data against `schema/causalmosaic.yaml`.

The CAMO schema is the source of truth for graph structure, enums and
constraints. This module runs the LinkML validator on a graph file and returns
the problems as strings, so the CLI and pipeline can report them in the
evidence trace.
"""

from __future__ import annotations

import os
from pathlib import Path

import hbreader
from linkml.validator import validate_file


def _install_path_safe_hbreader() -> None:
    """Stop hbreader from mistaking file paths that contain a quote for YAML text.

    LinkML reads files through ``hbreader``, whose default test treats any string
    containing ``'`` or ``"`` as inline data. A checkout under a folder such as
    ``Moji's Hackathon`` then fails to load the schema (and LinkML's own bundled
    ``types.yaml``) with ``ValueError: Unexpected type <path>``. A single-line
    string naming an existing file is a path, whatever characters it contains.
    """
    original = hbreader.default_str_tester
    if getattr(original, "_path_safe", False):
        return

    def tester(s: str) -> bool:
        if "\n" not in s and "\r" not in s and os.path.isfile(s):
            return False
        return original(s)

    tester._path_safe = True  # type: ignore[attr-defined]
    for fn in (hbreader.hbread, hbreader.hbopen, hbreader.detect_type):
        defaults = fn.__defaults__ or ()
        fn.__defaults__ = tuple(tester if d is original else d for d in defaults)
    hbreader.default_str_tester = tester


_install_path_safe_hbreader()

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
