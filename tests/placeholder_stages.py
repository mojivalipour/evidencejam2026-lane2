"""No-LLM placeholder stages for tests.

Retrieval is Haystack BM25 over the evidence records. Reasoning and verification
are placeholders: every draft is `no_evidence` with confidence 0.0. The production
pipeline uses `causa_nostra.pipeline.stages.build_agent_stages` instead.
"""

from __future__ import annotations

from causa_nostra.answers.models import AnswerEntry
from causa_nostra.graph.store import GraphStore
from causa_nostra.pipeline.stages import Stages
from causa_nostra.pipeline.types import Question, QuestionFrame, Verification
from causa_nostra.retrieval.pipeline import RetrievalPipeline
from causa_nostra.retrieval.records import EvidenceRecord, build_records

_STOPWORDS = frozenset(
    "what does do is the of on to in and or a an how effect effects affect affects "
    "by for with from which are be it its".split()
)


def _terms(text: str) -> list[str]:
    words = (w.strip("?.,:;()\"'").lower() for w in text.split())
    return sorted({w for w in words if w and w not in _STOPWORDS})


def build_placeholder_stages(store: GraphStore, top_k: int = 20) -> Stages:
    records_all = build_records(store)
    retrieval = RetrievalPipeline(records_all, top_k=top_k)

    def understand(question: Question) -> QuestionFrame:
        return QuestionFrame(
            question_id=question.id,
            intervention=question.question,
            outcome="",
            search_terms=_terms(question.question),
        )

    def retrieve(frame: QuestionFrame) -> list[EvidenceRecord]:
        return retrieval.retrieve(frame)

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

    def verify(question: Question, frame: QuestionFrame, draft: AnswerEntry, records: list[EvidenceRecord]) -> Verification:
        return Verification(passed=True, criteria={}, problems=[])

    return Stages(understand=understand, retrieve=retrieve, reason=reason, verify=verify)
