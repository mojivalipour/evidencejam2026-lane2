"""Stage interfaces and the production builder.

`Stages` is the injectable contract that `pipeline.run` and `pipeline.batch` depend on.
`build_agent_stages` wires the three LLM agents and retrieval into it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from causa_nostra.agents.base import ChatClient
from causa_nostra.agents.question_processor import QuestionProcessor
from causa_nostra.agents.reasoner import Reasoner
from causa_nostra.agents.verifier import Verifier
from causa_nostra.answers.models import AnswerEntry
from causa_nostra.graph.store import GraphStore
from causa_nostra.pipeline.types import Question, QuestionFrame, Verification
from causa_nostra.retrieval.pipeline import RetrievalPipeline
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


def build_agent_stages(store: GraphStore, client: ChatClient, top_k: int = 20) -> Stages:
    retrieval = RetrievalPipeline(build_records(store), top_k=top_k)
    processor = QuestionProcessor(client)
    reasoner = Reasoner(client)
    verifier = Verifier(client)

    return Stages(
        understand=processor.run,
        retrieve=retrieval.retrieve,
        reason=reasoner.run,
        verify=verifier.run,
    )


if __name__ == "__main__":
    # Demo: answer one question end to end with the three agents and the real endpoint.
    from causa_nostra.llm.client import LLMClient, LLMSettings
    from causa_nostra.pipeline.run import answer_question

    store = GraphStore.from_json("causal_graph.json")
    client = LLMClient(LLMSettings.from_env())
    try:
        question = Question(id="demo", question="What does burning do to non-native grass cover?")
        entry = answer_question(question, store, build_agent_stages(store, client))
        print(entry.model_dump_json(indent=2))
    finally:
        client.close()
