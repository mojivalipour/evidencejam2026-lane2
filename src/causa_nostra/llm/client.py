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


class LLMResponseError(RuntimeError):
    """The endpoint answered but the reply has no usable content."""


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
        timeout: float = 60.0,
        retries: int = 1,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._model = settings.model
        self._retries = retries
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
        thinking: bool = True,
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            # Qwen3 reasoning can be switched off for short structured tasks (vLLM chat template flag).
            "chat_template_kwargs": {"enable_thinking": thinking},
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        data = self._post_with_retry(payload)

        usage = data.get("usage") or {}
        choice = data["choices"][0]
        content = (choice.get("message") or {}).get("content")
        log.info(
            "llm usage prompt_tokens=%s completion_tokens=%s finish_reason=%s",
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            choice.get("finish_reason"),
        )
        if not content:
            # Reasoning models can spend the whole max_tokens budget on reasoning_content.
            raise LLMResponseError(
                f"empty content (finish_reason={choice.get('finish_reason')}); raise max_tokens"
            )
        return ChatResult(
            text=content,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )

    def close(self) -> None:
        self._http.close()

    def _post_with_retry(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST once, plus `retries` more times on timeouts, connection errors, or 5xx replies."""
        attempts = self._retries + 1
        for attempt in range(1, attempts + 1):
            try:
                response = self._http.post("/chat/completions", json=payload)
                if response.status_code >= 500 and attempt < attempts:
                    log.warning("llm server error %s; retrying (%d/%d)", response.status_code, attempt, attempts)
                    continue
                response.raise_for_status()
                return response.json()
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt == attempts:
                    raise
                log.warning("llm transport error %s; retrying (%d/%d)", type(exc).__name__, attempt, attempts)
        raise AssertionError("unreachable")


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
