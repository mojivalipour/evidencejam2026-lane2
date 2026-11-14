"""Data passed between pipeline stages. Each stage sees one question only."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Question(BaseModel):
    """One input question. Hidden-set entries carry only `id` and `question`."""

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)


class QuestionFrame(BaseModel):
    """Output of Understand: what the question asks, in graph terms."""

    question_id: str
    intervention: str
    outcome: str
    context: list[str] = Field(default_factory=list)
    search_terms: list[str] = Field(default_factory=list)  # candidates, not equivalences


class Verification(BaseModel):
    """Output of Verify. Semantic judgements come from the verifier LLM."""

    passed: bool
    criteria: dict[str, str] = Field(default_factory=dict)  # criterion -> pass|partial|fail
    problems: list[str] = Field(default_factory=list)
    supporting_edge_ids: list[str] = Field(default_factory=list)


if __name__ == "__main__":
    # Demo: show the intermediate models as JSON.
    question = Question(id="demo", question="What does X do to Y?")
    frame = QuestionFrame(question_id=question.id, intervention="X", outcome="Y", search_terms=["x"])
    verification = Verification(passed=False, criteria={"citation_support": "fail"}, problems=["demo"])
    for model in (question, frame, verification):
        print(model.model_dump_json())
