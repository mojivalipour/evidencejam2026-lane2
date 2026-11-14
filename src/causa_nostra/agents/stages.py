"""Wire the three agents and retrieval into the pipeline's `Stages`.

Plug-in point only: `pipeline.run.answer_question` is unchanged. Each agent is
independent and testable on its own; this module just connects them.
"""

from __future__ import annotations

from causa_nostra.agents.base import ChatClient
from causa_nostra.agents.question_processor import QuestionProcessor
from causa_nostra.agents.reasoner import Reasoner
from causa_nostra.agents.verifier import Verifier
from causa_nostra.graph.store import GraphStore
from causa_nostra.pipeline.stages import Stages
from causa_nostra.retrieval.pipeline import RetrievalPipeline
from causa_nostra.retrieval.records import build_records


def build_agent_stages(store: GraphStore, client: ChatClient, top_k: int = 20) -> Stages:
    retrieval = RetrievalPipeline(build_records(store), top_k=top_k)
    processor = QuestionProcessor(client)
    reasoner = Reasoner(client)
    verifier = Verifier(client)

    def reason(question, frame, records, feedback):
        return reasoner.run(question, frame, records, feedback)

    def verify(question, draft, records):
        return verifier.run(question, draft, records)

    return Stages(
        understand=processor.run,
        retrieve=retrieval.retrieve,
        reason=reason,
        verify=verify,
    )


if __name__ == "__main__":
    # Demo: answer one question end to end with the three agents and the real endpoint.
    from causa_nostra.llm.client import LLMClient, LLMSettings
    from causa_nostra.pipeline.run import answer_question
    from causa_nostra.pipeline.types import Question

    store = GraphStore.from_json("causal_graph.json")
    client = LLMClient(LLMSettings.from_env())
    try:
        question = Question(id="demo", question="What does burning do to non-native grass cover?")
        entry = answer_question(question, store, build_agent_stages(store, client))
        print(entry.model_dump_json(indent=2))
    finally:
        client.close()
