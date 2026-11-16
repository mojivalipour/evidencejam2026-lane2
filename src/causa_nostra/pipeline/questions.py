"""Load practice or hidden question files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from causa_nostra.pipeline.types import Question


def load_questions(path: str | Path) -> list[Question]:
    data: Any = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    items = data.get("questions") if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError(f"{path}: expected a list of questions under 'questions'")

    questions = [Question.model_validate(item) for item in items]
    seen: set[str] = set()
    for question in questions:
        if question.id in seen:
            raise ValueError(f"{path}: duplicate question id {question.id}")
        seen.add(question.id)
    return questions


if __name__ == "__main__":
    # Demo: load the practice questions and list their ids.
    for question in load_questions("questions.yaml"):
        print(f"{question.id}: {question.question[:70]}")
