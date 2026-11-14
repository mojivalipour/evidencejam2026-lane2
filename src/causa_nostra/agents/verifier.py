"""Agent 3: verification.

The LLM returns a per-criterion judgement (pass / partial / fail), short problem
notes, and supporting edge IDs. The pass/fail decision and the weighted score are
computed here in Python, not by the model.

Rubric weights follow the project instructions: evidence match 30%, direction and
causal language 30%, citation support 25%, search coverage 15%.
"""

from __future__ import annotations

from causa_nostra.agents.base import Agent, ChatClient
from causa_nostra.answers.models import AnswerEntry
from causa_nostra.pipeline.types import Question, Verification
from causa_nostra.retrieval.records import EvidenceRecord

CRITERIA_WEIGHTS: dict[str, float] = {
    "evidence_match": 0.30,
    "direction_and_causal_language": 0.30,
    "citation_support": 0.25,
    "search_coverage": 0.15,
}
# Criteria whose failure requires correction regardless of other scores.
REQUIRED_CRITERIA = ("direction_and_causal_language", "citation_support")
VALID_JUDGEMENTS = {"pass", "partial", "fail"}
POINTS = {"pass": 1.0, "partial": 0.5, "fail": 0.0}

SYSTEM_PROMPT = """You verify a draft answer against the evidence records it cites.
Judge each criterion as "pass", "partial", or "fail":
- "evidence_match": the cited records match both the intervention and the outcome, and the requested context.
- "direction_and_causal_language": the direction, null findings, and causal wording match the records.
- "citation_support": every factual claim in the answer is supported by a cited record.
- "search_coverage": the records are plausibly enough to answer the question, or a no_evidence answer is justified.

Return ONLY a JSON object with these keys:
- "criteria": object mapping each of the four criterion names above to "pass", "partial", or "fail".
- "problems": list of short strings describing each concrete problem. Empty list if none.
- "supporting_edge_ids": list of edge_ids from the records that bear on the question."""


def rubric_score(criteria: dict[str, str]) -> float:
    """Weighted score in [0, 1] from per-criterion judgements. Missing criteria score 0."""
    return sum(weight * POINTS.get(criteria.get(name, "fail"), 0.0) for name, weight in CRITERIA_WEIGHTS.items())


def _judgements(raw: object) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    return {
        name: str(value).lower()
        for name, value in raw.items()
        if name in CRITERIA_WEIGHTS and str(value).lower() in VALID_JUDGEMENTS
    }


class Verifier(Agent):
    """Structured judgement over short inputs: reasoning is switched off for speed and energy."""

    def __init__(self, client: ChatClient, *, max_tokens: int = 1024) -> None:
        super().__init__(client, max_tokens=max_tokens, thinking=False)

    def run(self, question: Question, draft: AnswerEntry, records: list[EvidenceRecord]) -> Verification:
        by_id = {r.edge_id: r for r in records}
        cited_text = "\n".join(f"[{e}] {by_id[e].text}" for e in draft.edge_ids if e in by_id) or "(none)"
        user = (
            f"Question: {question.question}\n"
            f"Draft status: {draft.status}\n"
            f"Draft answer: {draft.answer}\n"
            f"Draft direction: {draft.direction}\n"
            f"Draft certainty: {draft.certainty}\n"
            f"Cited records:\n{cited_text}\n"
        )
        data = self._ask_json(SYSTEM_PROMPT, user)

        criteria = _judgements(data.get("criteria"))
        raw_problems = data.get("problems")
        problems = [str(p).strip() for p in raw_problems if str(p).strip()] if isinstance(raw_problems, list) else []
        raw_supporting = data.get("supporting_edge_ids")
        supporting = [e for e in raw_supporting if isinstance(e, str) and e in by_id] if isinstance(raw_supporting, list) else []

        # Missing or invalid judgements count as fail, so the verifier cannot pass by omission.
        for name in CRITERIA_WEIGHTS:
            criteria.setdefault(name, "fail")
        # Any failed criterion that requires correction, or failed evidence match, fails the draft.
        failed = [name for name, value in criteria.items() if value == "fail"]
        passed = not any(name in failed for name in (*REQUIRED_CRITERIA, "evidence_match"))
        if not passed and not problems:
            problems = [f"criteria failed: {', '.join(failed)}"]
        return Verification(
            passed=passed,
            criteria=criteria,
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
        verification = Verifier(client).run(Question(id="demo", question="demo"), draft, [record])
        print(verification.model_dump_json(indent=2))
        print(f"rubric score: {rubric_score(verification.criteria):.2f}")
    finally:
        client.close()
