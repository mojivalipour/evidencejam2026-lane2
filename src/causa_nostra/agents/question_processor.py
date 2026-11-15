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
MAX_SIDE_TERMS = 10

SYSTEM_PROMPT = """You extract the structure of a causal question about ecology.
The question asks what one intervention or condition does to one outcome.

Return ONLY a JSON object with these keys:
- "intervention": the broad intervention or condition, as a short noun phrase.
- "outcome": the broad outcome and measured attribute, as a short noun phrase.
- "context": list of context constraints stated explicitly in the question. Use [] if none. Do not invent constraints.
- "intervention_terms": up to 10 alternative words or phrases for the INTERVENTION only (synonyms, singular/plural, common variants). Do not include outcome terms.
- "outcome_terms": up to 10 alternative words or phrases for the OUTCOME only (synonyms, singular/plural, common variants). Do not include intervention terms.

These are search candidates only. Do not answer the question. Do not add information that is not in the question."""

REFINE_SYSTEM_PROMPT = """A keyword search for a causal ecology question found no evidence in a graph.
Propose broader or different search terms for the SAME intervention and the SAME outcome.

Return ONLY a JSON object with these keys:
- "intervention_terms": up to 10 new words or phrases for the intervention (broader category, related practice, other spelling).
- "outcome_terms": up to 10 new words or phrases for the outcome (broader category, related measured attribute).

Do not repeat terms already tried. Do not change the intervention or outcome. Do not answer the question."""


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


def _side_terms(intervention_raw: object, outcome_raw: object) -> tuple[list[str], list[str]]:
    """Per-side synonyms. A term listed for both sides cannot tell them apart, so it is dropped from both."""
    intervention = _clean_list(intervention_raw)[:MAX_SIDE_TERMS]
    outcome = _clean_list(outcome_raw)[:MAX_SIDE_TERMS]
    shared = {t.lower() for t in intervention} & {t.lower() for t in outcome}
    return (
        [t for t in intervention if t.lower() not in shared],
        [t for t in outcome if t.lower() not in shared],
    )


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

        intervention_terms, outcome_terms = _side_terms(data.get("intervention_terms"), data.get("outcome_terms"))
        # `search_terms` is the combined candidate list for display and verification.
        combined = _clean_list([*_clean_list(data.get("search_terms")), *intervention_terms, *outcome_terms])
        return QuestionFrame(
            question_id=question.id,
            intervention=intervention,
            outcome=outcome,
            context=_clean_list(data.get("context")),
            search_terms=combined[:MAX_SEARCH_TERMS],
            intervention_terms=intervention_terms,
            outcome_terms=outcome_terms,
        )

    def refine(self, question: Question, frame: QuestionFrame) -> QuestionFrame:
        """Broaden the search terms after a failed search. Returns the frame unchanged if no new term is found."""
        tried = [*frame.intervention_terms, *frame.outcome_terms, *frame.search_terms]
        user = (
            f"Question: {question.question}\n"
            f"Intervention: {frame.intervention}\n"
            f"Outcome: {frame.outcome}\n"
            f"Terms already tried: {', '.join(tried) or 'none'}"
        )
        data = self._ask_json(REFINE_SYSTEM_PROMPT, user)
        known = {t.lower() for t in tried}
        new_intervention, new_outcome = _side_terms(data.get("intervention_terms"), data.get("outcome_terms"))
        new_intervention = [t for t in new_intervention if t.lower() not in known]
        new_outcome = [t for t in new_outcome if t.lower() not in known]
        if not new_intervention and not new_outcome:
            return frame
        intervention_terms = _merge(frame.intervention_terms, new_intervention)
        outcome_terms = _merge(frame.outcome_terms, new_outcome)
        return QuestionFrame(
            question_id=frame.question_id,
            intervention=frame.intervention,
            outcome=frame.outcome,
            context=frame.context,
            search_terms=_merge(frame.search_terms, [*new_intervention, *new_outcome])[:MAX_SEARCH_TERMS],
            intervention_terms=intervention_terms,
            outcome_terms=outcome_terms,
        )


def _merge(existing: list[str], extra: list[str]) -> list[str]:
    """Existing terms first, then new ones, without case-insensitive duplicates, capped per side."""
    merged = _clean_list([*existing, *extra])
    return merged[: 2 * MAX_SIDE_TERMS]


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
        f"intervention terms ({len(frame.intervention_terms)}):",
    ]
    lines += [f"  - {term}" for term in frame.intervention_terms]
    lines.append(f"outcome terms ({len(frame.outcome_terms)}):")
    lines += [f"  - {term}" for term in frame.outcome_terms]
    lines.append(f"search_terms ({len(frame.search_terms)}):")
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
