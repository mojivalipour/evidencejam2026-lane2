"""Agent 3: verification.

The LLM judges each rubric sub-item (pass / partial / fail) and returns short problem
notes and supporting edge IDs. Category judgements, gates, the score, and `passed` are
computed in `causa_nostra.answers.rubric`, which is the single source of truth.
"""

from __future__ import annotations

from causa_nostra.agents.base import Agent, AgentOutputError, ChatClient
from causa_nostra.answers.models import AnswerEntry
from causa_nostra.answers.rubric import (
    ANSWERED_QUESTIONS,
    JUDGEMENT_CHOICES,
    NO_EVIDENCE_GATE_QUESTIONS,
    NO_EVIDENCE_QUESTIONS,
    RubricResult,
    all_item_texts,
    evaluate_answered,
    evaluate_no_evidence,
    item_texts,
    normalise,
    normalise_item,
    rubric_score,
)
from causa_nostra.pipeline.types import Question, QuestionFrame, Verification
from causa_nostra.retrieval.pipeline import search_query
from causa_nostra.retrieval.records import EvidenceRecord, structured_text

__all__ = [
    "NO_EVIDENCE_SYSTEM_PROMPT",
    "SYSTEM_PROMPT",
    "Verifier",
    "build_system_prompt",
    "rubric_score",
]


def build_system_prompt(answered: bool | None = None) -> str:
    """Verifier prompt. `answered=None` lists all 22 sub-items; True or False lists only that draft's own items.

    Asking only the applicable items keeps the reply short, which keeps verification fast.
    """
    choices = ", ".join(f'"{choice}"' for choice in JUDGEMENT_CHOICES)
    groups = {
        "Answered-draft items": ANSWERED_QUESTIONS,
        "Missing-evidence items": {**NO_EVIDENCE_QUESTIONS, "absence_gate": {"N6": NO_EVIDENCE_GATE_QUESTIONS["N6"]}},
    }
    if answered is True:
        groups.pop("Missing-evidence items")
    elif answered is False:
        groups.pop("Answered-draft items")
    own_ids = all_item_texts() if answered is None else item_texts(answered)
    sections = "\n\n".join(
        f"{group}:\n" + "\n\n".join(
            f"{category}:\n" + "\n".join(f'- "{item}": {text}' for item, text in items.items())
            for category, items in categories.items()
        )
        for group, categories in groups.items()
    )
    item_ids = ", ".join(f'"{item}"' for item in own_ids)
    guidance = (
        'Mark an item "irrelevant" when it does not apply to this draft. For an answered draft, the '
        "Missing-evidence items do not apply. For a no_evidence draft, the Answered-draft items do not apply, "
        'and the citation items C1-C4 do not apply when it cites no edges. Mark A4 "irrelevant" when the question '
        "requested no species, ecosystem, or comparator context. Every other item must be pass, partial, or fail."
    )
    return (
        "You verify a draft answer against the evidence records and search log provided.\n"
        f"Judge each item as {choices}. Use only the records and search log; do not add outside knowledge.\n"
        f"{guidance}\n\n"
        f"{sections}\n\n"
        "Answer EVERY item id listed below. Omitting an id is not allowed.\n"
        "Return ONLY a JSON object with these keys:\n"
        f'- "items": object mapping each of these ids to {choices}: {item_ids}.\n'
        '- "reasons": object mapping each of the same ids to a reason of at most 15 words, '
        "citing edge_ids where relevant. Every id needs a non-empty reason.\n"
        '- "problems": list of short strings describing each concrete problem. Empty list if none.\n'
        '- "supporting_edge_ids": list of edge_ids that directly answer BOTH the intervention and the outcome '
        "(and the requested context). Exclude edges that relate to only one side of the question."
    )


SYSTEM_PROMPT = build_system_prompt()
ANSWERED_SYSTEM_PROMPT = build_system_prompt(True)
NO_EVIDENCE_SYSTEM_PROMPT = build_system_prompt(False)


class Verifier(Agent):
    """Structured judgement over short inputs: reasoning is switched off for speed and energy."""

    def __init__(self, client: ChatClient, *, max_tokens: int = 4096) -> None:
        # 22 items with reasons need more than 1024 tokens; a truncated reply cannot be parsed.
        super().__init__(client, max_tokens=max_tokens, thinking=False)

    def _ask(self, system: str, user: str) -> dict:
        """One JSON call, retried once if the reply cannot be parsed (e.g. truncated output)."""
        try:
            return self._ask_json(system, user)
        except AgentOutputError:
            return self._ask_json(system, user)

    def run(
        self,
        question: Question,
        frame: QuestionFrame,
        draft: AnswerEntry,
        records: list[EvidenceRecord],
    ) -> Verification:
        # Status selects both the prompt and the rubric: missing-evidence drafts never get answered items.
        answered = draft.status == "answered"
        system = ANSWERED_SYSTEM_PROMPT if answered else NO_EVIDENCE_SYSTEM_PROMPT

        by_id = {r.edge_id: r for r in records}
        cited_text = "\n".join(
            f"[{e}]\n{structured_text(by_id[e])}" for e in draft.edge_ids if e in by_id
        ) or "(none)"
        # Retrieved records are listed as one short line each: the verifier only needs them to spot
        # which edges bear on the question. Full text is sent only for cited records.
        retrieved_text = "\n".join(
            f"[{r.edge_id}] {r.subject_label[:160]} -> {r.object_label[:160]}" for r in records
        ) or "(none)"
        search_log = (
            f"Executed query: {search_query(frame)}\n"
            f"Alternative terms searched: {', '.join(frame.search_terms) or '(none)'}\n"
            f"Retrieved edge_ids (top k, ranked): {', '.join(r.edge_id for r in records) or '(none)'}"
        )
        user = (
            f"Question: {question.question}\n"
            f"Processed intervention: {frame.intervention}\n"
            f"Processed outcome: {frame.outcome}\n"
            f"Processed context: {', '.join(frame.context) if frame.context else '(none)'}\n"
            f"Draft status: {draft.status}\n"
            f"Draft answer: {draft.answer}\n"
            f"Draft direction: {draft.direction}\n"
            f"Draft certainty: {draft.certainty}\n"
            f"Search log:\n{search_log}\n\n"
            f"Cited records:\n{cited_text}\n\n"
            f"Retrieved records (top k, structured fields):\n{retrieved_text}\n"
        )
        data = self._ask(system, user)

        expected = item_texts(answered)
        own = expected
        data, gaps = self._complete(system, user, data, expected)
        raw_items = data.get("items")
        raw_items = raw_items if isinstance(raw_items, dict) else {}
        raw_reasons = data.get("reasons")
        raw_reasons = raw_reasons if isinstance(raw_reasons, dict) else {}
        # An item with no judgement or no reason, even after the retry, counts as fail.
        # Only the draft's own rubric is scored; the other items are recorded but not scored.
        items = {
            item: "fail" if item in gaps else normalise_item(raw_items.get(item), item, own) for item in expected
        }
        item_reasons = {
            item: (str(raw_reasons.get(item) or "").strip() or "no reason returned") for item in expected
        }
        scored = {item: items[item] for item in own}
        result: RubricResult = evaluate_answered(scored) if answered else evaluate_no_evidence(scored)
        raw_problems = data.get("problems")
        problems = [str(p).strip() for p in raw_problems if str(p).strip()] if isinstance(raw_problems, list) else []
        if gaps:
            problems.insert(0, f"verifier left items unanswered: {', '.join(gaps)}")
        raw_supporting = data.get("supporting_edge_ids")
        # Only edges that matched both the intervention and the outcome count as supporting.
        # A one-sided candidate (e.g. the right outcome, a different intervention) must not block a valid no_evidence.
        both_sided = {r.edge_id for r in records if {"subject", "object"} <= set(r.retrieval_ranks)}
        supporting = (
            [e for e in raw_supporting if isinstance(e, str) and e in by_id and e in both_sided]
            if isinstance(raw_supporting, list)
            else []
        )

        passed = result.passed
        if not answered and supporting:
            # A missing-evidence draft cannot pass when the verifier itself names retrieved edges that bear on the question.
            passed = False
            problems.append(f"no_evidence draft, but retrieved edges bear on the question: {', '.join(supporting)}")
        if not passed and not problems:
            problems = [f"gated items failed: {', '.join(result.failed_gates)}"]
        return Verification(
            passed=passed,
            criteria=result.criteria,
            items=items,
            item_reasons=item_reasons,
            problems=problems,
            supporting_edge_ids=supporting,
            score=result.score,
        )

    @staticmethod
    def _gaps(data: dict, expected: dict[str, str]) -> list[str]:
        """Ids with no judgement or no reason in the verifier's reply."""
        items = data.get("items") if isinstance(data.get("items"), dict) else {}
        reasons = data.get("reasons") if isinstance(data.get("reasons"), dict) else {}
        return [
            item
            for item in expected
            if not str(items.get(item) or "").strip() or not str(reasons.get(item) or "").strip()
        ]

    def _complete(self, system: str, user: str, data: dict, expected: dict[str, str]) -> tuple[dict, list[str]]:
        """Ask once more for missing ids. Returns the reply to use and any ids still missing."""
        missing = self._gaps(data, expected)
        if not missing:
            return data, []
        retry = (
            f"{user}\n\nYour previous reply left these ids without a judgement or reason: "
            f"{', '.join(missing)}. Return the complete JSON object with every id answered."
        )
        data = self._ask(system, retry)
        return data, self._gaps(data, expected)


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
