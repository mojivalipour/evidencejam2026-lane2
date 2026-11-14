"""OpenAI-compatible chat client for the Qwen endpoint.

Configuration comes only from the environment:
    BRINE_URL, BRINE_MODEL, BRINE_API_KEY

The API key is never logged or included in reprs. Token usage is logged so
energy can be estimated per question.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Mapping
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
        backoff: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if retries < 0:
            raise ValueError("retries must be >= 0")
        self._model = settings.model
        self._retries = retries
        self._backoff = backoff
        self._sleep = sleep
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
        thinking: bool = False,
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
        try:
            choice = data["choices"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError("reply has no choices") from exc
        if not isinstance(choice, Mapping):
            raise LLMResponseError("reply choice is not an object")
        content = (choice.get("message") or {}).get("content")
        if content is not None and not isinstance(content, str):
            raise LLMResponseError(f"content has unexpected type {type(content).__name__}")
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

    def __enter__(self) -> LLMClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _post_with_retry(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST once, plus `retries` more times on timeouts, connection errors, 429, or 5xx replies.

        Other 4xx replies are not retried. Waits `backoff * 2**(attempt-1)` seconds between attempts.
        """
        attempts = self._retries + 1
        for attempt in range(1, attempts + 1):
            retryable_reason: str | None = None
            try:
                response = self._http.post("/chat/completions", json=payload)
                status = response.status_code
                if (status == 429 or status >= 500) and attempt < attempts:
                    retryable_reason = f"http {status}"
                else:
                    response.raise_for_status()
                    try:
                        return response.json()
                    except ValueError as exc:
                        raise LLMResponseError("reply is not valid JSON") from exc
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt == attempts:
                    raise
                retryable_reason = type(exc).__name__
            log.warning("llm %s; retrying (%d/%d)", retryable_reason, attempt, attempts)
            self._sleep(self._backoff * 2 ** (attempt - 1))
        raise AssertionError("unreachable")


if __name__ == "__main__":
    # Connectivity check. Makes one small request to the endpoint named in the environment.
    logging.basicConfig(level=logging.INFO)
    try:
        settings = LLMSettings.from_env()
    except LLMConfigError as exc:
        raise SystemExit(f"{exc}. Set them in .env or the shell, then retry.")
    client = LLMClient(settings)
    messages = [{"role": "user", "content": "Reply with the single word OK."}]
    try:
        # Thinking is disabled so the short budget is spent on the answer, not on reasoning.
        print(f"query: {messages}")
        result = client.chat(messages, max_tokens=64, thinking=False)
        print(f"response: {result.text!r}")
        print(f"model {settings.model} replied: {result.text.strip()!r}")
    finally:
        client.close()
