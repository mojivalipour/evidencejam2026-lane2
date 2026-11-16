"""Live check: QuestionProcessor over every question in questions.yaml.

Makes real LLM calls, so it is skipped unless BRINE_URL, BRINE_MODEL and
BRINE_API_KEY are set. Run with output visible:

    set -a; source .env; set +a
    uv run pytest tests/test_question_processor_questions.py -s
"""

import os
from pathlib import Path

import pytest

from causa_nostra.agents.question_processor import MAX_SEARCH_TERMS, QuestionProcessor, format_frame
from causa_nostra.llm.client import LLMClient, LLMSettings
from causa_nostra.pipeline.questions import load_questions

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = ROOT / "questions.yaml"
REQUIRED_ENV = ("BRINE_URL", "BRINE_MODEL", "BRINE_API_KEY")

pytestmark = pytest.mark.skipif(
    not all(os.environ.get(name) for name in REQUIRED_ENV),
    reason="BRINE_* environment variables not set",
)


@pytest.fixture(scope="module")
def client():
    client = LLMClient(LLMSettings.from_env())
    yield client
    client.close()


@pytest.mark.parametrize("question", load_questions(QUESTIONS_PATH), ids=lambda q: q.id)
def test_question_processor_frame_for_each_question(client, question):
    frame = QuestionProcessor(client).run(question)
    print(format_frame(question, frame))

    assert frame.question_id == question.id
    assert frame.intervention.strip()
    assert frame.outcome.strip()
    assert 0 < len(frame.search_terms) <= MAX_SEARCH_TERMS
    assert all(term.strip() for term in frame.search_terms)
