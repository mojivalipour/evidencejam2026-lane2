"""A draft that never passes verification keeps its citations at lower confidence; it is not relabelled."""

from causa_nostra.graph.models import Edge, Graph, Node
from causa_nostra.graph.store import GraphStore
from causa_nostra.pipeline.run import answer_question
from causa_nostra.pipeline.stages import Stages
from causa_nostra.pipeline.types import Question, QuestionFrame, Verification
from causa_nostra.answers.models import AnswerEntry


def _store() -> GraphStore:
    return GraphStore(
        Graph(
            graph_id="t",
            nodes=[Node(id="S"), Node(id="O")],
            edges=[Edge(id="e1", subject="S", predicate="causes", object="O", causal_language="direct_causal")],
        )
    )


def test_failed_verification_keeps_citations_at_lower_confidence():
    def reason(question, frame, records, feedback):
        return AnswerEntry(
            id=question.id,
            status="answered",
            answer="Decrease.",
            direction="decrease",
            confidence=0.9,
            certainty="low",
            edge_ids=["e1"],
        )

    def verify(question, frame, draft, records):
        return Verification(passed=False, score=0.4, problems=["direction wording too strong"])

    stages = Stages(
        understand=lambda q: QuestionFrame(question_id=q.id, intervention="S", outcome="O"),
        retrieve=lambda frame: [],
        reason=reason,
        verify=verify,
    )
    entry = answer_question(Question(id="Q1", question="q"), _store(), stages, max_revisions=1)

    assert entry.status == "answered"
    assert entry.edge_ids == ["e1"]
    assert entry.direction == "decrease"
    assert entry.confidence == 0.4
