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
from causa_nostra.pipeline.types import Question

REPO_ROOT = Path(__file__).resolve().parents[3]


class DraftRejected(RuntimeError):
    """A draft still cites unknown edges after the allowed revisions."""


def answer_question(
    question: Question,
    store: GraphStore,
    stages: Stages,
    *,
    max_revisions: int = 1,
) -> AnswerEntry:
    frame = stages.understand(question)
    records = stages.retrieve(frame)

    draft = stages.reason(question, frame, records, [])
    for attempt in range(max_revisions + 1):
        # Deterministic checks first: cheap, and they catch hallucinated citations.
        problems = check_entry(draft, store)
        if not problems:
            verification = stages.verify(question, frame, draft, records)
            if verification.passed:
                return draft
            problems = verification.problems or ["verification failed"]
        if attempt == max_revisions:
            break
        draft = stages.reason(question, frame, records, problems)

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
