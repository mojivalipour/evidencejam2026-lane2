"""Stage tracing for one question.

INFO: one line per stage (start, result summary), always shown.
DEBUG: the input and output of each stage, clipped, shown only in debug mode.

Debug mode is enabled by `causa-nostra run --debug` or `CAUSA_DEBUG=1`.
Trace lines go to stderr, so the answers file is never mixed with them.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from pydantic import BaseModel

TRACE = logging.getLogger("causa_nostra.trace")
MAX_PAYLOAD_CHARS = 4000
_NOISY_LIBRARIES = ("httpx", "haystack", "urllib3")


def clip(text: str, limit: int = MAX_PAYLOAD_CHARS) -> str:
    """Shorten long text, keeping its start and saying how much was cut."""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [+{len(text) - limit} chars]"


def payload(value: Any) -> str:
    """One-line text for a stage input or output: JSON for models, repr otherwise."""
    if isinstance(value, BaseModel):
        return clip(value.model_dump_json())
    if isinstance(value, (list, tuple)) and value and all(isinstance(v, BaseModel) for v in value):
        return clip("[" + ", ".join(v.model_dump_json() for v in value) + "]")
    return clip(repr(value))


def stage_info(question_id: str, stage: str, message: str, *args: object) -> None:
    TRACE.info("[%s] %s: " + message, question_id, stage, *args)


def stage_debug(question_id: str, stage: str, label: str, value: Any) -> None:
    if TRACE.isEnabledFor(logging.DEBUG):
        TRACE.debug("[%s] %s %s: %s", question_id, stage, label, payload(value))


def debug_requested(flag: bool = False) -> bool:
    return flag or os.environ.get("CAUSA_DEBUG") == "1"


def configure_logging(debug: bool = False) -> None:
    """Send trace output to stderr. INFO by default, DEBUG when debug is requested."""
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in _NOISY_LIBRARIES:
        logging.getLogger(name).setLevel(logging.WARNING)


if __name__ == "__main__":
    # Demo: trace one fake stage at INFO and DEBUG on a small model. No LLM is involved.
    from causa_nostra.pipeline.types import QuestionFrame

    configure_logging(debug=True)
    frame = QuestionFrame(question_id="demo", intervention="burning", outcome="grass cover")
    stage_info("demo", "understand", "intervention=%r outcome=%r", frame.intervention, frame.outcome)
    stage_debug("demo", "understand", "output", frame)
    stage_debug("demo", "understand", "long text", "x" * (MAX_PAYLOAD_CHARS + 10))
