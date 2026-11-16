"""Shared base for the Qwen-backed agents.

Each agent makes one small, deterministic LLM call (temperature 0) and parses a
JSON object from the reply. Parsing failures raise `AgentOutputError`, which the
batch runner contains per question.
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Protocol

from causa_nostra.llm.client import ChatResult

SHOW_PROMPTS_ENV = "CAUSA_SHOW_PROMPTS"


class AgentOutputError(ValueError):
    """The model reply could not be parsed as the expected JSON object."""


class ChatClient(Protocol):
    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        thinking: bool = True,
    ) -> ChatResult: ...


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_json_object(text: str) -> dict[str, Any]:
    """Extract one JSON object from a model reply, tolerating code fences and prose."""
    stripped = _FENCE.sub("", text.strip()).strip()
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start == -1 or end <= start:
            raise AgentOutputError("no JSON object in model output") from None
        try:
            data = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError as exc:
            raise AgentOutputError(f"invalid JSON in model output: {exc}") from exc
    if not isinstance(data, dict):
        raise AgentOutputError("expected a JSON object in model output")
    return data


class Agent:
    """Base class: holds the chat client and provides one JSON-returning call."""

    def __init__(self, client: ChatClient, *, max_tokens: int = 4096, thinking: bool = True) -> None:
        self._client = client
        self._max_tokens = max_tokens
        self._thinking = thinking

    def _ask_json(self, system: str, user: str) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        result = self._client.chat(
            messages,
            temperature=0.0,
            max_tokens=self._max_tokens,
            thinking=self._thinking,
        )
        if os.environ.get(SHOW_PROMPTS_ENV):
            print(
                f"--- {type(self).__name__} request (thinking={self._thinking}, max_tokens={self._max_tokens})\n"
                f"[system]\n{system}\n[user]\n{user}\n"
                f"--- {type(self).__name__} raw reply\n{result.text}\n---",
                file=sys.stderr,
            )
        return parse_json_object(result.text)


if __name__ == "__main__":
    # Demo: one real call to the endpoint named in BRINE_URL / BRINE_MODEL / BRINE_API_KEY.
    from causa_nostra.llm.client import LLMClient, LLMSettings

    client = LLMClient(LLMSettings.from_env())
    try:
        data = Agent(client, max_tokens=256)._ask_json("Reply only with a JSON object.", 'Return {"ok": true}.')
        print(f"parsed reply: {data}")
    finally:
        client.close()
