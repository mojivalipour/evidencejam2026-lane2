"""Three agents for the reasoning pipeline.

- `QuestionProcessor`: question -> QuestionFrame (understand).
- `Reasoner`: frame + evidence records -> draft AnswerEntry (reason).
- `Verifier`: draft + records -> Verification (verify), with a Python-computed rubric score.

Each agent makes one small LLM call at temperature 0 and is tested with a fake client.
"""

from causa_nostra.agents.base import Agent, AgentOutputError, parse_json_object
from causa_nostra.agents.question_processor import QuestionProcessor
from causa_nostra.agents.reasoner import Reasoner
from causa_nostra.agents.verifier import Verifier, rubric_score

__all__ = [
    "Agent",
    "AgentOutputError",
    "QuestionProcessor",
    "Reasoner",
    "Verifier",
    "parse_json_object",
    "rubric_score",
]
