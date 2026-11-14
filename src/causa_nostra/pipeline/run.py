"""Pipeline orchestration: one independent run per question, then batch-level checks.

No conversation history is shared between questions. A verification failure is
never relabelled as `no_evidence`; it is recorded and the draft is revised at
most `max_revisions` times.
"""

from __future__ import annotations

from collections.abc import Iterable

from causa_nostra.answers.models import AnswerEntry, AnswersFile, TEAM_NAME
from causa_nostra.answers.validate import check_entry, check_file
from causa_nostra.graph.store import GraphStore
from causa_nostra.pipeline.stages import Stages
from causa_nostra.pipeline.types import Question


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
            verification = stages.verify(question, draft, records)
            if verification.passed:
                return draft
            problems = verification.problems or ["verification failed"]
        if attempt == max_revisions:
            break
        draft = stages.reason(question, frame, records, problems)

    raise DraftRejected(f"{question.id}: unresolved after revisions: {'; '.join(problems)}")


def run_pipeline(
    questions: Iterable[Question],
    store: GraphStore,
    stages: Stages,
    *,
    max_revisions: int = 1,
) -> AnswersFile:
    questions = list(questions)
    entries = [answer_question(q, store, stages, max_revisions=max_revisions) for q in questions]
    answers = AnswersFile(team=TEAM_NAME, answers=entries)

    problems = check_file(answers, store, (q.id for q in questions))
    if problems:
        raise ValueError("invalid answers: " + "; ".join(problems))
    return answers


if __name__ == "__main__":
    # Demo: run the whole orchestration on the first three practice questions.
    from causa_nostra.graph.store import GraphStore
    from causa_nostra.pipeline.questions import load_questions
    from causa_nostra.pipeline.stages import build_placeholder_stages

    store = GraphStore.from_json("causal_graph.json")
    questions = load_questions("questions.yaml")[:3]
    answers = run_pipeline(questions, store, build_placeholder_stages(store))
    print(answers.model_dump_json(indent=2))
