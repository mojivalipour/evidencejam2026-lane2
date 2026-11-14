"""OpenAI-compatible chat client for the Qwen endpoint.

Configuration comes only from the environment:
    BRINE_URL, BRINE_MODEL, BRINE_API_KEY

The API key is never logged or included in reprs. Token usage is logged so
energy can be estimated per question.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import httpx
from pydantic import BaseModel

log = logging.getLogger(__name__)

ENV_URL = "BRINE_URL"
ENV_MODEL = "BRINE_MODEL"
ENV_KEY = "BRINE_API_KEY"


class LLMConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMSettings:
    url: str
    model: str
    api_key: str = field(repr=False)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> LLMSettings:
        source = os.environ if env is None else env
        missing = [name for name in (ENV_URL, ENV_MODEL, ENV_KEY) if not source.get(name)]
        if missing:
            raise LLMConfigError(f"missing environment variables: {', '.join(missing)}")
        return cls(url=source[ENV_URL], model=source[ENV_MODEL], api_key=source[ENV_KEY])


class ChatResult(BaseModel):
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class LLMClient:
    def __init__(
        self,
        settings: LLMSettings,
        *,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._model = settings.model
        self._http = httpx.Client(
            base_url=settings.url.rstrip("/"),
            headers={"Authorization": f"Bearer {settings.api_key}"},
            timeout=timeout,
            transport=transport,
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        response = self._http.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()

        usage = data.get("usage") or {}
        log.info(
            "llm usage prompt_tokens=%s completion_tokens=%s",
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
        )
        return ChatResult(
            text=data["choices"][0]["message"]["content"],
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )

    def close(self) -> None:
        self._http.close()


if __name__ == "__main__":
    # Connectivity check. Makes one small request to the endpoint named in the environment.
    logging.basicConfig(level=logging.INFO)
    try:
        settings = LLMSettings.from_env()
    except LLMConfigError as exc:
        raise SystemExit(f"{exc}. Set them in .env or the shell, then retry.")
    client = LLMClient(settings)
    try:
        result = client.chat([{"role": "user", "content": "Reply with the single word OK."}], max_tokens=8)
        print(f"model {settings.model} replied: {result.text.strip()!r}")
    finally:
        client.close()
