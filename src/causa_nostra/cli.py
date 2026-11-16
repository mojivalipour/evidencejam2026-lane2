"""Command-line entry point.

    causa-nostra check-graph [--graph causal_graph.json]
    causa-nostra run QUESTIONS OUT [--graph causal_graph.json]

`run.sh` will wrap `run` as `./run.sh questions.yaml answers.json`.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from causa_nostra.camo_rules.checks import check_graph
from causa_nostra.camo_rules.schema import validate_graph_file
from causa_nostra.graph.store import GraphStore
from causa_nostra.llm.client import LLMClient, LLMSettings
from causa_nostra.pipeline.batch import run_batch
from causa_nostra.pipeline.questions import load_questions
from causa_nostra.pipeline.stages import build_agent_stages
from causa_nostra.pipeline.trace import configure_logging, debug_requested

DEFAULT_GRAPH = "causal_graph.json"


def _cmd_check_graph(args: argparse.Namespace) -> int:
    store = GraphStore.from_json(args.graph)
    print(f"nodes={len(store.graph.nodes)} edges={len(store.graph.edges)}")
    schema_problems = validate_graph_file(args.graph)
    if schema_problems:
        print(f"LinkML schema problems: {len(schema_problems)}")
        for message in schema_problems[:20]:
            print(f"  {message}")
    else:
        print("LinkML schema: valid")
    issues = check_graph(store)
    if not issues:
        print("no invariant violations")
    for rule, count in Counter(i.rule for i in issues).most_common():
        print(f"{rule}: {count}")
    return 1 if (schema_problems or issues) else 0


def _cmd_run(args: argparse.Namespace) -> int:
    configure_logging(debug_requested(args.debug))
    store = GraphStore.from_json(args.graph)
    questions = load_questions(args.questions)
    client = LLMClient(LLMSettings.from_env())
    try:
        answers = run_batch(questions, store, build_agent_stages(store, client), args.output)
    finally:
        client.close()
    print(f"wrote {len(answers.answers)} answers to {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="causa-nostra")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check-graph", help="validate graph invariants")
    check.add_argument("--graph", default=DEFAULT_GRAPH)
    check.set_defaults(func=_cmd_check_graph)

    run = sub.add_parser("run", help="answer a question file")
    run.add_argument("questions")
    run.add_argument("output")
    run.add_argument("--graph", default=DEFAULT_GRAPH)
    run.add_argument("--debug", action="store_true", help="also log each stage's input and output (or set CAUSA_DEBUG=1)")
    run.set_defaults(func=_cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
