"""The competition output contract (answers.json).

Field names and enums match the Evidence Jam format exactly. Cross-field rules
for `answered` and `no_evidence` are enforced here, so an invalid entry cannot
be written.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

TEAM_NAME = "CAUSA NOSTRA"

Status = Literal["answered", "no_evidence"]
Direction = Literal["increase", "decrease", "no_effect", "mixed", "unclear"]
Certainty = Literal["high", "moderate", "low", "very_low"]


class AnswerEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    status: Status
    answer: str = Field(min_length=1)
    direction: Direction | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    certainty: Certainty | None = None
    edge_ids: list[str] = Field(default_factory=list)
    searched_for: str | None = None
    extras: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _status_consistency(self) -> AnswerEntry:
        if self.status == "answered":
            if self.direction is None:
                raise ValueError(f"{self.id}: answered entries need a direction")
            if self.certainty is None:
                raise ValueError(f"{self.id}: answered entries need a certainty")
            if not self.edge_ids:
                raise ValueError(f"{self.id}: answered entries need edge_ids")
        else:
            if self.direction is not None or self.certainty is not None:
                raise ValueError(f"{self.id}: no_evidence entries must have null direction and certainty")
            if self.edge_ids:
                raise ValueError(f"{self.id}: no_evidence entries must have empty edge_ids")
            if not self.searched_for:
                raise ValueError(f"{self.id}: no_evidence entries must describe searched_for")
        return self


class AnswersFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    team: Literal["CAUSA NOSTRA"] = TEAM_NAME
    answers: list[AnswerEntry]

    @model_validator(mode="after")
    def _unique_ids(self) -> AnswersFile:
        seen: set[str] = set()
        for entry in self.answers:
            if entry.id in seen:
                raise ValueError(f"duplicate answer id: {entry.id}")
            seen.add(entry.id)
        return self


if __name__ == "__main__":
    # Demo: one valid answered entry and one valid no_evidence entry, printed as JSON.
    sample = AnswersFile(
        answers=[
            AnswerEntry(
                id="demo-1",
                status="answered",
                answer="Demo answer.",
                direction="decrease",
                confidence=0.6,
                certainty="low",
                edge_ids=["camo:edge_demo"],
            ),
            AnswerEntry(
                id="demo-2",
                status="no_evidence",
                answer="Demo: nothing found.",
                confidence=0.5,
                searched_for="demo search terms",
            ),
        ]
    )
    print(sample.model_dump_json(indent=2))
