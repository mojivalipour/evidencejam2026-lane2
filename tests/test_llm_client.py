"""Offline tests for LLMClient. No network or credentials needed."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from causa_nostra.llm.client import (
    ChatResult,
    LLMClient,
    LLMConfigError,
    LLMResponseError,
    LLMSettings,
)

SETTINGS = LLMSettings(url="https://llm.example/v1", model="qwen-test", api_key="secret-key")
OK_BODY = {
    "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 1},
}


def make_client(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    retries: int = 1,
) -> tuple[LLMClient, list[float]]:
    sleeps: list[float] = []
    client = LLMClient(
        SETTINGS,
        retries=retries,
        backoff=1.0,
        sleep=sleeps.append,
        transport=httpx.MockTransport(handler),
    )
    return client, sleeps


def ok_handler(_: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=OK_BODY)


def test_thinking_is_disabled_by_default() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=OK_BODY)

    client, _ = make_client(handler)
    client.chat([{"role": "user", "content": "hi"}])
    assert seen[0]["chat_template_kwargs"] == {"enable_thinking": False}


def test_thinking_can_be_enabled_explicitly() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=OK_BODY)

    client, _ = make_client(handler)
    client.chat([{"role": "user", "content": "hi"}], thinking=True)
    assert seen[0]["chat_template_kwargs"] == {"enable_thinking": True}


def test_successful_chat_returns_text_and_usage() -> None:
    client, _ = make_client(ok_handler)
    result = client.chat([{"role": "user", "content": "hi"}])
    assert result == ChatResult(text="OK", prompt_tokens=5, completion_tokens=1)


@pytest.mark.parametrize("status", [429, 500, 503])
def test_retryable_status_is_retried_with_backoff(status: int) -> None:
    replies = [httpx.Response(status), httpx.Response(200, json=OK_BODY)]
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return replies.pop(0)

    client, sleeps = make_client(handler, retries=1)
    assert client.chat([{"role": "user", "content": "hi"}]).text == "OK"
    assert calls == 2
    assert sleeps == [1.0]


def test_client_error_is_not_retried() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, json={"error": "bad request"})

    client, sleeps = make_client(handler, retries=3)
    with pytest.raises(httpx.HTTPStatusError):
        client.chat([{"role": "user", "content": "hi"}])
    assert calls == 1
    assert sleeps == []


def test_persistent_server_error_raises_after_retries() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(502)

    client, sleeps = make_client(handler, retries=2)
    with pytest.raises(httpx.HTTPStatusError):
        client.chat([{"role": "user", "content": "hi"}])
    assert calls == 3
    assert sleeps == [1.0, 2.0]


def test_timeout_is_retried_then_raised() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectTimeout("timed out")

    client, sleeps = make_client(handler, retries=1)
    with pytest.raises(httpx.ConnectTimeout):
        client.chat([{"role": "user", "content": "hi"}])
    assert calls == 2
    assert sleeps == [1.0]


def test_negative_retries_rejected() -> None:
    with pytest.raises(ValueError):
        LLMClient(SETTINGS, retries=-1)


def test_reply_without_choices_raises_response_error() -> None:
    client, _ = make_client(lambda _: httpx.Response(200, json={"choices": []}))
    with pytest.raises(LLMResponseError, match="no choices"):
        client.chat([{"role": "user", "content": "hi"}])


def test_choice_that_is_not_an_object_raises_response_error() -> None:
    client, _ = make_client(lambda _: httpx.Response(200, json={"choices": ["oops"]}))
    with pytest.raises(LLMResponseError, match="not an object"):
        client.chat([{"role": "user", "content": "hi"}])


def test_non_string_content_raises_response_error() -> None:
    body = {"choices": [{"message": {"content": ["x"]}, "finish_reason": "stop"}]}
    client, _ = make_client(lambda _: httpx.Response(200, json=body))
    with pytest.raises(LLMResponseError, match="unexpected type"):
        client.chat([{"role": "user", "content": "hi"}])


def test_empty_content_raises_response_error() -> None:
    body = {"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}
    client, _ = make_client(lambda _: httpx.Response(200, json=body))
    with pytest.raises(LLMResponseError, match="empty content"):
        client.chat([{"role": "user", "content": "hi"}])


def test_invalid_json_raises_response_error() -> None:
    client, _ = make_client(lambda _: httpx.Response(200, text="<html>not json</html>"))
    with pytest.raises(LLMResponseError, match="not valid JSON"):
        client.chat([{"role": "user", "content": "hi"}])


def test_api_key_is_not_shown_in_repr() -> None:
    assert "secret-key" not in repr(SETTINGS)


def test_from_env_reports_missing_variables() -> None:
    with pytest.raises(LLMConfigError, match="BRINE_API_KEY"):
        LLMSettings.from_env({"BRINE_URL": "https://x", "BRINE_MODEL": "m"})


def test_from_env_reads_all_variables() -> None:
    settings = LLMSettings.from_env(
        {"BRINE_URL": "https://x", "BRINE_MODEL": "m", "BRINE_API_KEY": "k"}
    )
    assert (settings.url, settings.model, settings.api_key) == ("https://x", "m", "k")
