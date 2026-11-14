"""Stage interfaces. Each stage is injectable, so it can be tested or replaced alone.

Implementations that call Qwen belong in their own modules and are wired in
`build_default_stages`. Until they exist, that function raises.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from causa_nostra.answers.models import AnswerEntry
from causa_nostra.graph.store import GraphStore
from causa_nostra.pipeline.types import Question, QuestionFrame, Verification
from causa_nostra.retrieval.bm25 import BM25Index
from causa_nostra.retrieval.records import EvidenceRecord, build_records

# Understand: question -> frame
Understand = Callable[[Question], QuestionFrame]
# Retrieve: frame -> candidate records (may broaden the search once)
Retrieve = Callable[[QuestionFrame], list[EvidenceRecord]]
# Reason: question, frame, records, feedback from a failed verification -> draft
Reason = Callable[[Question, QuestionFrame, list[EvidenceRecord], list[str]], AnswerEntry]
# Verify: question, draft, records -> verification
Verify = Callable[[Question, AnswerEntry, list[EvidenceRecord]], Verification]


@dataclass(frozen=True)
class Stages:
    understand: Understand
    retrieve: Retrieve
    reason: Reason
    verify: Verify


_STOPWORDS = frozenset(
    "what does do is the of on to in and or a an how effect effects affect affects "
    "by for with from which are be it its".split()
)


def _terms(text: str) -> list[str]:
    words = (w.strip("?.,:;()\"'").lower() for w in text.split())
    return sorted({w for w in words if w and w not in _STOPWORDS})


def build_placeholder_stages(store: GraphStore, top_k: int = 20) -> Stages:
    """Runnable end-to-end placeholder: no LLM.

    Retrieval is Haystack BM25 over the evidence records. Reasoning and
    verification are placeholders: every draft is marked as unreliable
    (confidence 0.0) and says that reasoning is not implemented yet.
    """
    records_all = build_records(store)
    by_id = {r.edge_id: r for r in records_all}
    index = BM25Index(records_all)

    def understand(question: Question) -> QuestionFrame:
        return QuestionFrame(
            question_id=question.id,
            intervention=question.question,
            outcome="",
            search_terms=_terms(question.question),
        )

    def retrieve(frame: QuestionFrame) -> list[EvidenceRecord]:
        query = " ".join([frame.intervention, frame.outcome, *frame.search_terms]).strip()
        return [by_id[hit.edge_id] for hit in index.search(query, top_k=top_k)]

    def reason(
        question: Question,
        frame: QuestionFrame,
        records: list[EvidenceRecord],
        feedback: list[str],
    ) -> AnswerEntry:
        return AnswerEntry(
            id=question.id,
            status="no_evidence",
            answer=(
                "Placeholder: reasoning stage not implemented. "
                f"{len(records)} candidate edge(s) retrieved, none assessed."
            ),
            direction=None,
            confidence=0.0,
            certainty=None,
            edge_ids=[],
            searched_for=f"keyword search over edge records for terms: {', '.join(frame.search_terms)}",
        )

    def verify(question: Question, draft: AnswerEntry, records: list[EvidenceRecord]) -> Verification:
        return Verification(passed=True, criteria={}, problems=[])

    return Stages(understand=understand, retrieve=retrieve, reason=reason, verify=verify)


if __name__ == "__main__":
    # Demo: run the placeholder stages on the first three practice questions.
    from causa_nostra.graph.store import GraphStore
    from causa_nostra.pipeline.questions import load_questions

    store = GraphStore.from_json("causal_graph.json")
    stages = build_placeholder_stages(store)
    for question in load_questions("questions.yaml")[:3]:
        frame = stages.understand(question)
        records = stages.retrieve(frame)
        draft = stages.reason(question, frame, records, [])
        print(f"{question.id}: {len(records)} candidates -> {draft.status}")
