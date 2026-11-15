"""Pipeline orchestration: one independent run per question, then batch-level checks.

No conversation history is shared between questions. A verification failure is
never relabelled as `no_evidence`; it is recorded and the draft is revised at
most `max_revisions` times.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from causa_nostra.answers.models import AnswerEntry, AnswersFile, TEAM_NAME
from causa_nostra.answers.validate import check_entry, check_file
from causa_nostra.graph.store import GraphStore
from causa_nostra.pipeline.stages import Stages
from causa_nostra.pipeline.trace import stage_debug, stage_info
from causa_nostra.pipeline.types import Question, Verification

REPO_ROOT = Path(__file__).resolve().parents[3]


class DraftRejected(RuntimeError):
    """A draft still cites unknown edges after the allowed revisions."""


def _verification_feedback(verification: Verification) -> list[str]:
    """Feedback for the reasoner's revision: what failed, which edges bear on the question, and why."""
    feedback = list(verification.problems) or ["verification failed"]
    failed = [f"{name}={value}" for name, value in verification.criteria.items() if value != "pass"]
    if failed:
        feedback.append("Failed criteria: " + ", ".join(failed))
    if verification.supporting_edge_ids:
        feedback.append(
            "Retrieved edges that bear on the question: "
            + ", ".join(verification.supporting_edge_ids)
            + ". Use them in the answer, or state why each one does not answer the question."
        )
    return feedback


def answer_question(
    question: Question,
    store: GraphStore,
    stages: Stages,
    *,
    max_revisions: int = 1,
) -> AnswerEntry:
    qid = question.id
    stage_info(qid, "understand", "start")
    stage_debug(qid, "understand", "input", question)
    frame = stages.understand(question)
    stage_info(qid, "understand", "intervention=%r outcome=%r", frame.intervention, frame.outcome)
    stage_debug(qid, "understand", "output", frame)

    stage_info(qid, "retrieve", "start")
    stage_debug(qid, "retrieve", "input", frame)
    records = stages.retrieve(frame)
    top = ", ".join(r.edge_id for r in records[:5]) or "none"
    stage_info(qid, "retrieve", "%d record(s), top: %s", len(records), top)
    stage_debug(qid, "retrieve", "output", records)

    feedback: list[str] = []
    for attempt in range(max_revisions + 1):
        stage_info(qid, "reason", "attempt %d, feedback items: %d", attempt + 1, len(feedback))
        stage_debug(qid, "reason", "input feedback", feedback)
        draft = stages.reason(question, frame, records, feedback)
        stage_info(
            qid, "reason", "status=%s direction=%s edges=%d", draft.status, draft.direction, len(draft.edge_ids)
        )
        stage_debug(qid, "reason", "output", draft)

        # Deterministic checks first: cheap, and they catch hallucinated citations.
        problems = check_entry(draft, store)
        if problems:
            stage_info(qid, "check", "failed: %d problem(s)", len(problems))
            stage_debug(qid, "check", "problems", problems)
        else:
            stage_info(qid, "check", "passed")
            stage_info(qid, "verify", "start")
            verification = stages.verify(question, frame, draft, records)
            stage_info(qid, "verify", "passed=%s score=%s", verification.passed, verification.score)
            stage_debug(qid, "verify", "output", verification)
            if verification.passed:
                # The accepted confidence is the deterministic rubric score, not the model's self-estimate.
                if verification.score is not None:
                    draft = draft.model_copy(update={"confidence": verification.score})
                stage_info(qid, "done", "accepted after %d attempt(s)", attempt + 1)
                return draft
            problems = _verification_feedback(verification)
        if attempt == max_revisions:
            break
        feedback = problems

    stage_info(qid, "done", "rejected after %d attempt(s)", max_revisions + 1)
    raise DraftRejected(f"{question.id}: unresolved after revisions: {'; '.join(problems)}")


def _unresolved_entry(question: Question, reason: str) -> AnswerEntry:
    """Entry for a question whose draft could not be verified.

    The challenge scores a missing entry as wrong at full confidence, so every
    input question must get an entry. Confidence 0.0 states that the graph is
    not known to lack evidence; the reason is recorded in the answer text.
    """
    return AnswerEntry(
        id=question.id,
        status="no_evidence",
        answer=f"Could not produce a verified answer: {reason}",
        direction=None,
        confidence=0.0,
        certainty=None,
        edge_ids=[],
        searched_for="pipeline failed before a verified answer was produced",
    )


def run_pipeline(
    questions: Iterable[Question],
    store: GraphStore,
    stages: Stages,
    *,
    max_revisions: int = 1,
) -> AnswersFile:
    questions = list(questions)
    entries: list[AnswerEntry] = []
    for question in questions:
        try:
            entries.append(answer_question(question, store, stages, max_revisions=max_revisions))
        except DraftRejected as exc:
            entries.append(_unresolved_entry(question, str(exc)))
    answers = AnswersFile(team=TEAM_NAME, answers=entries)

    problems = check_file(answers, store, (q.id for q in questions), repo_root=REPO_ROOT)
    if problems:
        raise ValueError("invalid answers: " + "; ".join(problems))
    return answers


if __name__ == "__main__":
    # Demo: run the whole orchestration on the first three practice questions.
    from causa_nostra.graph.store import GraphStore
    from causa_nostra.pipeline.questions import load_questions
    from causa_nostra.llm.client import LLMClient, LLMSettings
    from causa_nostra.pipeline.stages import build_agent_stages

    store = GraphStore.from_json("causal_graph.json")
    questions = load_questions("questions.yaml")[:3]
    client = LLMClient(LLMSettings.from_env())
    try:
        answers = run_pipeline(questions, store, build_agent_stages(store, client))
        print(answers.model_dump_json(indent=2))
    finally:
        client.close()
