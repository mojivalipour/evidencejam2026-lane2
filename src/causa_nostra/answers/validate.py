"""Validate answer-file fields and run deterministic pipeline consistency checks.

Answer fields are validated against the Pydantic output contract. Pipeline
checks additionally verify citations, question coverage, and extra-file paths.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

from pydantic import ValidationError

from causa_nostra.answers.models import AnswerEntry, AnswersFile
from causa_nostra.graph.store import GraphStore


def validate_answer_file(path: Path) -> list[str]:
    """Return answer-field validation errors from an answers.json file."""
    content = path.read_text(encoding="utf-8")
    try:
        AnswersFile.model_validate_json(content)
    except ValidationError as exc:
        return [
            f"{'.'.join(str(part) for part in error['loc']) or '<file>'}: {error['msg']}"
            for error in exc.errors(include_input=False)
        ]
    return []


def check_entry(entry: AnswerEntry, store: GraphStore) -> list[str]:
    return [
        f"{entry.id}: citation {edge_id} does not exist in the graph"
        for edge_id in entry.edge_ids
        if not store.has_edge(edge_id)
    ]


def check_extras(entry: AnswerEntry, repo_root: Path) -> list[str]:
    """Extras must be repo-relative paths to files that exist."""
    problems = []
    for extra in entry.extras:
        path = Path(extra)
        if path.is_absolute() or ".." in path.parts:
            problems.append(f"{entry.id}: extra {extra} must be a relative path inside the repo")
        elif not (repo_root / path).is_file():
            problems.append(f"{entry.id}: extra {extra} does not exist")
    return problems


def check_file(
    answers: AnswersFile,
    store: GraphStore,
    question_ids: Iterable[str],
    repo_root: Path | None = None,
) -> list[str]:
    expected = list(question_ids)
    expected_set = set(expected)
    returned = [entry.id for entry in answers.answers]
    returned_set = set(returned)

    problems = [f"missing answer for question {qid}" for qid in expected if qid not in returned_set]
    problems += [f"answer for unknown question {aid}" for aid in returned_set - expected_set]
    for entry in answers.answers:
        problems.extend(check_entry(entry, store))
        if repo_root is not None:
            problems.extend(check_extras(entry, repo_root))
    return problems


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate answer fields in an answers.json file.")
    parser.add_argument("answers_file", nargs="?", type=Path, default=Path("answers.json"))
    args = parser.parse_args()

    problems = validate_answer_file(args.answers_file)
    if problems:
        print(f"{args.answers_file}: invalid answer file")
        for problem in problems:
            print(problem)
        raise SystemExit(1)
    print(f"{args.answers_file}: answer fields are valid")
