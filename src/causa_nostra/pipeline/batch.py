"""Fault-tolerant batch runner.

Guarantees:
- Every input question gets exactly one entry, in input order.
- A failure in one question (exception, invalid citation, invalid extra) is
  contained: that question gets a valid fallback entry and the others continue.
- After every question the partial answers are written atomically, so a crash
  or interruption leaves a complete, parseable `answers.json` on disk.

Warnings are logged, never raised, so one bad entry cannot discard a batch.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

from causa_nostra.answers.models import AnswerEntry, AnswersFile, TEAM_NAME
from causa_nostra.answers.evidence import write_evidence
from causa_nostra.answers.validate import check_extras
from causa_nostra.graph.store import GraphStore
from causa_nostra.pipeline.run import REPO_ROOT, DraftRejected, answer_question
from causa_nostra.pipeline.stages import Stages
from causa_nostra.pipeline.types import Question

log = logging.getLogger(__name__)

MAX_REASON_CHARS = 300


def fallback_entry(question_id: str, reason: str) -> AnswerEntry:
    """Valid `no_evidence` entry for a question that could not be answered.

    The `searched_for` text says this is a pipeline failure, not a graph finding.
    """
    reason = reason.strip()[:MAX_REASON_CHARS] or "unknown error"
    return AnswerEntry(
        id=question_id,
        status="no_evidence",
        answer=f"Could not produce a verified answer (pipeline error): {reason}",
        direction=None,
        confidence=0.0,
        certainty=None,
        edge_ids=[],
        searched_for="not searched: pipeline error before a verified answer was produced",
    )


def sanitize_entry(entry: AnswerEntry, store: GraphStore, repo_root: Path) -> tuple[AnswerEntry, list[str]]:
    """Drop invalid citations and extras; downgrade to fallback if no valid citation remains.

    Returns the cleaned entry and a list of warnings describing each repair.
    """
    warnings: list[str] = []

    valid_edges = []
    for edge_id in entry.edge_ids:
        if store.has_edge(edge_id):
            valid_edges.append(edge_id)
        else:
            warnings.append(f"{entry.id}: dropped unknown citation {edge_id}")

    valid_extras = []
    for extra in entry.extras:
        probe = entry.model_copy(update={"extras": [extra]})
        problems = check_extras(probe, repo_root)
        if problems:
            warnings.extend(problems)
        else:
            valid_extras.append(extra)

    if entry.status == "answered" and not valid_edges:
        return fallback_entry(entry.id, "every cited edge was invalid"), warnings

    try:
        cleaned = AnswerEntry.model_validate(
            entry.model_dump() | {"edge_ids": valid_edges, "extras": valid_extras}
        )
    except ValueError as exc:
        warnings.append(f"{entry.id}: entry failed validation after repair: {exc}")
        return fallback_entry(entry.id, "entry failed validation after repair"), warnings
    return cleaned, warnings


def attach_evidence(entry: AnswerEntry, store: GraphStore, evidence_dir: Path, repo_root: Path) -> AnswerEntry:
    """Write the evidence page for an answered entry and list it in `extras`."""
    try:
        path = write_evidence(entry, store, evidence_dir, repo_root)
    except Exception as exc:  # the page is optional: never lose the answer over it
        log.warning("evidence page for %s not written: %s", entry.id, exc)
        return entry
    if path is None or path in entry.extras:
        return entry
    return entry.model_copy(update={"extras": [*entry.extras, path]})


def write_answers_atomic(path: str | Path, answers: AnswersFile) -> None:
    """Write `answers.json` via temp file + fsync + rename, so readers never see a partial file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=target.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(answers.model_dump_json(indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _answer_with_retry(
    question: Question,
    store: GraphStore,
    stages: Stages,
    repo_root: Path,
    max_revisions: int,
) -> tuple[AnswerEntry, list[str]]:
    """Run one question, retrying once on unexpected errors (e.g. transient LLM failures).

    `DraftRejected` is not retried: it is a deterministic verification outcome.
    """
    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            draft = answer_question(question, store, stages, max_revisions=max_revisions)
            return sanitize_entry(draft, store, repo_root)
        except DraftRejected:
            raise
        except Exception as exc:
            last_exc = exc
            log.warning("question %s attempt %d failed: %s", question.id, attempt + 1, exc)
    assert last_exc is not None
    raise last_exc


def run_batch(
    questions: Iterable[Question],
    store: GraphStore,
    stages: Stages,
    output_path: str | Path,
    *,
    repo_root: Path = REPO_ROOT,
    max_revisions: int = 1,
) -> AnswersFile:
    """Answer every question independently and checkpoint after each one."""
    questions = list(questions)
    entries: dict[str, AnswerEntry] = {}
    warnings: list[str] = []
    failed: list[str] = []
    # One evidence directory per output file, so two batches never overwrite each other's pages.
    output = Path(output_path)
    evidence_dir = output.parent / f"{output.stem}_evidence"

    for question in questions:
        try:
            entry, repairs = _answer_with_retry(question, store, stages, repo_root, max_revisions)
            warnings.extend(repairs)
        except Exception as exc:  # per-question isolation: one failure must not stop the batch
            log.exception("question %s failed", question.id)
            failed.append(question.id)
            entry = fallback_entry(question.id, f"{type(exc).__name__}: {exc}")
        entry = attach_evidence(entry, store, evidence_dir, repo_root)
        entries[question.id] = entry

        partial = AnswersFile(answers=[entries[q.id] for q in questions if q.id in entries])
        write_answers_atomic(output_path, partial)

    for message in warnings:
        log.warning(message)
    if failed:
        log.warning("pipeline failed for %d question(s): %s", len(failed), ", ".join(failed))

    answers = AnswersFile(team=TEAM_NAME, answers=[entries[q.id] for q in questions])
    write_answers_atomic(output_path, answers)
    return answers


if __name__ == "__main__":
    # Demo: run one question through the agents with the real endpoint, into a temp file.
    import tempfile

    from causa_nostra.graph.store import GraphStore
    from causa_nostra.llm.client import LLMClient, LLMSettings
    from causa_nostra.pipeline.questions import load_questions
    from causa_nostra.pipeline.stages import build_agent_stages

    store = GraphStore.from_json("causal_graph.json")
    questions = load_questions("questions.yaml")[:1]
    client = LLMClient(LLMSettings.from_env())
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "answers.json"
            answers = run_batch(questions, store, build_agent_stages(store, client), out)
            print(f"wrote {len(answers.answers)} answer(s) to a temp file; statuses: {[a.status for a in answers.answers]}")
    finally:
        client.close()
