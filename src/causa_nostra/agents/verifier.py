"""Agent 3: verification.

The LLM judges each rubric sub-item (pass / partial / fail) and returns short problem
notes and supporting edge IDs. Category judgements, gates, the score, and `passed` are
computed in `causa_nostra.answers.rubric`, which is the single source of truth.
"""

from __future__ import annotations

from causa_nostra.agents.base import Agent, ChatClient
from causa_nostra.answers.models import AnswerEntry
from causa_nostra.answers.rubric import (
    ANSWERED_QUESTIONS,
    IRRELEVANT_GUIDANCE,
    JUDGEMENT_CHOICES,
    evaluate_answered,
    rubric_score,
)
from causa_nostra.pipeline.types import Question, QuestionFrame, Verification
from causa_nostra.retrieval.records import EvidenceRecord

__all__ = ["SYSTEM_PROMPT", "Verifier", "build_system_prompt", "rubric_score"]


def build_system_prompt(questions: dict[str, dict[str, str]] = ANSWERED_QUESTIONS) -> str:
    """Build the verifier prompt from the rubric's sub-items and judgement choices."""
    choices = ", ".join(f'"{choice}"' for choice in JUDGEMENT_CHOICES)
    sections = "\n\n".join(
        f"{category}:\n" + "\n".join(f'- "{item}": {text}' for item, text in items.items())
        for category, items in questions.items()
    )
    item_ids = ", ".join(f'"{item}"' for category in questions.values() for item in category)
    return (
        "You verify a draft answer against the evidence records it cites.\n"
        f"Judge each item as {choices}. Use only the records; do not add outside knowledge.\n"
        f"{IRRELEVANT_GUIDANCE}\n\n"
        f"{sections}\n\n"
        "Return ONLY a JSON object with these keys:\n"
        f'- "items": object mapping each of these ids to {choices}: {item_ids}.\n'
        '- "problems": list of short strings describing each concrete problem. Empty list if none.\n'
        '- "supporting_edge_ids": list of edge_ids from the records that bear on the question.'
    )


SYSTEM_PROMPT = build_system_prompt()


class Verifier(Agent):
    """Structured judgement over short inputs: reasoning is switched off for speed and energy."""

    def __init__(self, client: ChatClient, *, max_tokens: int = 1024) -> None:
        super().__init__(client, max_tokens=max_tokens, thinking=False)

    def run(
        self,
        question: Question,
        frame: QuestionFrame,
        draft: AnswerEntry,
        records: list[EvidenceRecord],
    ) -> Verification:
        by_id = {r.edge_id: r for r in records}
        cited_text = "\n".join(f"[{e}] {by_id[e].text}" for e in draft.edge_ids if e in by_id) or "(none)"
        retrieved_text = "\n".join(f"[{r.edge_id}] {r.text}" for r in records) or "(none)"
        user = (
            f"Question: {question.question}\n"
            f"Processed intervention: {frame.intervention}\n"
            f"Processed outcome: {frame.outcome}\n"
            f"Processed context: {', '.join(frame.context) if frame.context else '(none)'}\n"
            f"Draft status: {draft.status}\n"
            f"Draft answer: {draft.answer}\n"
            f"Draft direction: {draft.direction}\n"
            f"Draft certainty: {draft.certainty}\n"
            f"Cited records:\n{cited_text}\n\n"
            f"Retrieved records (top k):\n{retrieved_text}\n"
        )
        data = self._ask_json(SYSTEM_PROMPT, user)

        raw_items = data.get("items")
        result = evaluate_answered(raw_items if isinstance(raw_items, dict) else {})
        raw_problems = data.get("problems")
        problems = [str(p).strip() for p in raw_problems if str(p).strip()] if isinstance(raw_problems, list) else []
        raw_supporting = data.get("supporting_edge_ids")
        supporting = [e for e in raw_supporting if isinstance(e, str) and e in by_id] if isinstance(raw_supporting, list) else []

        if not result.passed and not problems:
            problems = [f"gated items failed: {', '.join(result.failed_gates)}"]
        return Verification(
            passed=result.passed,
            criteria=result.criteria,
            problems=problems,
            supporting_edge_ids=supporting,
        )


if __name__ == "__main__":
    # Demo: judge a draft against the first graph record with the real endpoint.
    from causa_nostra.graph.store import GraphStore
    from causa_nostra.llm.client import LLMClient, LLMSettings
    from causa_nostra.retrieval.records import build_records

    record = build_records(GraphStore.from_json("causal_graph.json"))[0]
    draft = AnswerEntry(id="demo", status="answered", answer=f"{record.subject_label} {record.predicate} {record.object_label}.",
                        direction="unclear", confidence=0.3, certainty="very_low", edge_ids=[record.edge_id])
    client = LLMClient(LLMSettings.from_env())
    try:
        verification = Verifier(client).run(
            Question(id="demo", question="demo"),
            QuestionFrame(question_id="demo", intervention=record.subject_label, outcome=record.object_label),
            draft,
            [record],
        )
        print(verification.model_dump_json(indent=2))
        print(f"rubric score: {rubric_score(verification.criteria):.2f}")
    finally:
        client.close()
