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
from causa_nostra.retrieval.records import EvidenceRecord

REPO_ROOT = Path(__file__).resolve().parents[3]

# Highest confidence for a no_evidence answer when retrieval found candidate records.
NO_EVIDENCE_WITH_CANDIDATES_CAP = 0.5


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

    # A verified no_evidence draft held back while the reasoner re-checks the candidates.
    held_no_evidence: AnswerEntry | None = None

    draft = stages.reason(question, frame, records, [])
    for attempt in range(max_revisions + 1):
        # Deterministic checks first: cheap, and they catch hallucinated citations.
        problems = check_entry(draft, store)
        if not problems:
            verification = stages.verify(question, frame, draft, records)
            if verification.passed:
                # The accepted confidence is the deterministic rubric score, not the model's self-estimate.
                if verification.score is not None:
                    draft = draft.model_copy(update={"confidence": verification.score})
                if draft.status != "no_evidence" or not records:
                    return draft
                # no_evidence although candidate records matched both sides: look once more, then cap.
                capped = _cap_no_evidence(draft)
                if held_no_evidence is not None or attempt == max_revisions:
                    return capped
                held_no_evidence = capped
                problems = [_recheck_message(records)]
            else:
                problems = verification.problems or ["verification failed"]
        if attempt == max_revisions:
            break
        draft = stages.reason(question, frame, records, problems)

    if held_no_evidence is not None:
        return held_no_evidence  # the re-check did not produce a verified answer
    raise DraftRejected(f"{question.id}: unresolved after revisions: {'; '.join(problems)}")


def _cap_no_evidence(draft: AnswerEntry) -> AnswerEntry:
    """Cap confidence of a no_evidence answer given while candidate records exist.

    The verifier can pass a too-narrow search; with candidates on both sides the
    absence claim is less certain, so it never goes out at full confidence.
    """
    return draft.model_copy(update={"confidence": min(draft.confidence, NO_EVIDENCE_WITH_CANDIDATES_CAP)})


def _recheck_message(records: list[EvidenceRecord]) -> str:
    return (
        f"The draft says no_evidence, but {len(records)} retrieved record(s) match both the "
        "intervention and the outcome side of the search. Check every record against each part "
        "of the question. If any record answers any part, return status \"answered\" for that part "
        "and say which parts have no evidence. Keep no_evidence only if no record matches any part."
    )


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
