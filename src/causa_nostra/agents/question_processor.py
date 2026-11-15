"""Agent 1: question processing.

Turns one question into a `QuestionFrame`: the intervention or condition, the
outcome, explicit context constraints, and alternative search terms. Search terms
are candidates for retrieval, not established equivalences, and no context
restriction is invented.
"""

from __future__ import annotations

from causa_nostra.agents.base import Agent, AgentOutputError, ChatClient
from causa_nostra.pipeline.types import Question, QuestionFrame

MAX_SEARCH_TERMS = 15

SYSTEM_PROMPT = """You extract the structure of a causal question about ecology.
The question asks what one intervention or condition does to one outcome.

Return ONLY a JSON object with these keys:
- "intervention": the broad intervention or condition, as a short noun phrase.
- "outcome": the broad outcome and measured attribute, as a short noun phrase.
- "context": list of context constraints stated explicitly in the question. Use [] if none. Do not invent constraints.
- "search_terms": list of up to 15 alternative words or phrases that could label the intervention or outcome in a scientific paper (synonyms, singular/plural, common variants). These are search candidates only.

Do not answer the question. Do not add information that is not in the question."""


def _clean_list(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    seen: list[str] = []
    for value in values:
        if isinstance(value, str):
            text = value.strip()
            if text and text.lower() not in (s.lower() for s in seen):
                seen.append(text)
    return seen


class QuestionProcessor(Agent):
    """Short structured extraction: reasoning is switched off for speed and energy."""

    def __init__(self, client: ChatClient, *, max_tokens: int = 1024) -> None:
        super().__init__(client, max_tokens=max_tokens, thinking=False)

    def run(self, question: Question) -> QuestionFrame:
        data = self._ask_json(SYSTEM_PROMPT, f"Question: {question.question}")

        intervention = str(data.get("intervention", "")).strip()
        outcome = str(data.get("outcome", "")).strip()
        if not intervention or not outcome:
            raise AgentOutputError("question processor did not extract both intervention and outcome")

        search_terms = _clean_list(data.get("search_terms"))[:MAX_SEARCH_TERMS]
        return QuestionFrame(
            question_id=question.id,
            intervention=intervention,
            outcome=outcome,
            context=_clean_list(data.get("context")),
            search_terms=search_terms,
        )


def format_frame(question: Question, frame: QuestionFrame) -> str:
    """Readable input/output block for terminal review."""
    width = 72
    lines = [
        "=" * width,
        f"ID:            {question.id}",
        f"INPUT:         {question.question}",
        "-" * width,
        f"intervention:  {frame.intervention}",
        f"outcome:       {frame.outcome}",
        f"context:       {', '.join(frame.context) if frame.context else '(none)'}",
        f"search_terms ({len(frame.search_terms)}):",
    ]
    lines += [f"  - {term}" for term in frame.search_terms]
    lines.append("=" * width)
    return "\n".join(lines)


if __name__ == "__main__":
    # Demo: extract the frame for one question with the real endpoint.
    from causa_nostra.llm.client import LLMClient, LLMSettings

    client = LLMClient(LLMSettings.from_env())
    try:
        question = Question(id="demo", question="What does burning do to non-native grass cover?")
        print(format_frame(question, QuestionProcessor(client).run(question)))
    finally:
        client.close()
