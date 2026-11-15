"""Agent 2: reasoning.

Given the question, its frame and retrieved evidence records, drafts one answer
that cites only edges from the supplied records. It does not chain edges or infer
new causal effects. Verifier feedback from a failed check is passed back in on
revision.
"""

from __future__ import annotations

import re

from pydantic import ValidationError

from causa_nostra.agents.base import Agent, ChatClient
from causa_nostra.answers.models import AnswerEntry
from causa_nostra.pipeline.types import Question, QuestionFrame
from causa_nostra.retrieval.pipeline import search_query
from causa_nostra.retrieval.records import EvidenceRecord

_MODEL_KEYS = ("status", "answer", "direction", "confidence", "certainty", "edge_ids", "searched_for")

NO_EVIDENCE_ANSWER = "The graph has no evidence answering this intervention–outcome question."

SYSTEM_PROMPT = """You answer a question using ONLY the evidence records provided. The graph is the source of truth.

Interpretation rules:
- The intervention and the outcome must both match the record. Related evidence is not a direct answer.
- Read the direction from the record's predicate and relation sign. Do not infer direction from the predicate alone.
- A record marked NULL RESULT is a tested null (direction "no_effect"), not missing evidence.
- Distinguish correlation or association from causal language. Do not present associations as causal.
- Keep conflicting records. If studies disagree, use direction "mixed".
- Do not chain edges to infer a new effect. Do not use outside knowledge.
- Cite only edge_ids that appear in the records. Cite the edges that support the answer.
- If the question has several parts (several interventions, outcomes or examples), answer each part the records cover and state plainly which parts have no matching evidence. Use status "answered" if any part is answered; use "no_evidence" only if no record matches any part.

Return ONLY a JSON object with these keys:
- "status": "answered" if matching evidence exists, otherwise "no_evidence".
- "answer": one to three short plain-language sentences. After each claim, put the supporting edge_id in square brackets, e.g. "Burning reduced grass cover [camo:edge_x]."
- "direction": "increase", "decrease", "no_effect", "mixed", or "unclear" if status is answered; otherwise null.
- "confidence": number from 0 to 1, your probability that the answer matches the graph.
- "certainty": "high", "moderate", "low", or "very_low" if answered; otherwise null.
- "edge_ids": list of cited edge_ids if answered; otherwise [].
- "searched_for": if no_evidence, a short description of what was searched; otherwise null."""


def format_records(records: list[EvidenceRecord], limit: int) -> str:
    lines = [f"[{r.edge_id}] {r.text}" for r in records[:limit]]
    return "\n".join(lines) if lines else "(no records retrieved)"


def _mentions(answer: str, edge_id: str) -> bool:
    """True when `edge_id` appears as a whole token (so `e1` does not match inside `e10`)."""
    return re.search(rf"(?<![\w:.-]){re.escape(edge_id)}(?![\w:.-])", answer) is not None


def with_citations(answer: str, edge_ids: list[str]) -> str:
    """Append any cited edge ID that the answer text does not already name.

    Deterministic, so every answered entry shows its citations even when the
    model leaves them out of the sentence.
    """
    missing = [edge_id for edge_id in edge_ids if not _mentions(answer, edge_id)]
    if not missing:
        return answer
    return f"{answer.rstrip()} [{'; '.join(missing)}]"


class Reasoner(Agent):
    """Evidence reasoning. Thinking is off for speed; re-enable if accuracy drops on practice questions."""

    def __init__(self, client: ChatClient, *, max_records: int = 40, max_tokens: int = 4096) -> None:
        super().__init__(client, max_tokens=max_tokens, thinking=False)
        self._max_records = max_records

    def run(
        self,
        question: Question,
        frame: QuestionFrame,
        records: list[EvidenceRecord],
        feedback: list[str],
    ) -> AnswerEntry:
        user = (
            f"Question: {question.question}\n"
            f"Intervention: {frame.intervention}\n"
            f"Outcome: {frame.outcome}\n"
            f"Context: {', '.join(frame.context) or 'none stated'}\n\n"
            f"Evidence records:\n{format_records(records, self._max_records)}\n"
        )
        if feedback:
            user += "\nThe previous draft failed verification. Fix these problems:\n- " + "\n- ".join(feedback)

        # One re-ask when the reply breaks the output contract (e.g. "answered" with no edge_ids),
        # with the validation error as feedback. A second invalid reply raises.
        for attempt in range(2):
            raw = self._ask_json(SYSTEM_PROMPT, user)
            try:
                entry = AnswerEntry.model_validate(self._entry_data(raw, question, frame))
                break
            except ValidationError as exc:
                if attempt == 1:
                    raise
                problems = "; ".join(err["msg"] for err in exc.errors())
                user += (
                    f"\n\nYour previous reply was invalid: {problems}. An \"answered\" reply needs a direction, "
                    "a certainty and at least one edge_id from the records. If no record supports any part "
                    "of the question, reply with status \"no_evidence\"."
                )
        if entry.status == "answered":
            entry = entry.model_copy(update={"answer": with_citations(entry.answer, entry.edge_ids)})
        return entry

    @staticmethod
    def _entry_data(raw: dict, question: Question, frame: QuestionFrame) -> dict:
        # AnswerEntry forbids unknown keys; drop anything extra the model adds (e.g. "reasoning").
        data = {key: raw[key] for key in _MODEL_KEYS if key in raw}
        data["id"] = question.id
        data["extras"] = []
        if not data.get("answer"):
            # Model often returns null for no_evidence; the contract needs a string.
            data["answer"] = NO_EVIDENCE_ANSWER
        if data.get("status") == "no_evidence":
            # The contract requires these empty for no_evidence; the model often fills them anyway.
            data.update(direction=None, certainty=None, edge_ids=[])
            if not data.get("searched_for"):
                data["searched_for"] = search_query(frame)
        return data


if __name__ == "__main__":
    # Demo: draft an answer over the first five graph records with the real endpoint.
    from causa_nostra.graph.store import GraphStore
    from causa_nostra.llm.client import LLMClient, LLMSettings
    from causa_nostra.retrieval.records import build_records

    records = build_records(GraphStore.from_json("causal_graph.json"))[:5]
    frame = QuestionFrame(question_id="demo", intervention=records[0].subject_label, outcome=records[0].object_label)
    client = LLMClient(LLMSettings.from_env())
    try:
        entry = Reasoner(client).run(Question(id="demo", question="demo"), frame, records, feedback=[])
        print(entry.model_dump_json(indent=2))
    finally:
        client.close()
