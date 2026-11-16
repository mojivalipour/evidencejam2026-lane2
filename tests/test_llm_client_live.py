"""Live tests against the configured Qwen endpoint.

These make real network calls. They are skipped unless BRINE_URL, BRINE_MODEL,
and BRINE_API_KEY are set in the environment. Run with:

    set -a; source .env; set +a
    uv run --frozen pytest tests/test_llm_client_live.py -q
"""

from __future__ import annotations

import os

import httpx
import pytest

from causa_nostra.llm.client import LLMClient, LLMConfigError, LLMSettings

_REQUIRED = ("BRINE_URL", "BRINE_MODEL", "BRINE_API_KEY")
_have_env = all(os.environ.get(name) for name in _REQUIRED)

live = pytest.mark.skipif(not _have_env, reason="BRINE_URL/BRINE_MODEL/BRINE_API_KEY not set")


# --- settings (no network) ------------------------------------------------


def test_from_env_reads_all_variables():
    s = LLMSettings.from_env({"BRINE_URL": "u", "BRINE_MODEL": "m", "BRINE_API_KEY": "k"})
    assert (s.url, s.model, s.api_key) == ("u", "m", "k")


def test_from_env_reports_all_missing_variables():
    with pytest.raises(LLMConfigError) as exc:
        LLMSettings.from_env({"BRINE_URL": "u"})
    assert "BRINE_MODEL" in str(exc.value)
    assert "BRINE_API_KEY" in str(exc.value)


def test_from_env_rejects_empty_values():
    with pytest.raises(LLMConfigError):
        LLMSettings.from_env({"BRINE_URL": "u", "BRINE_MODEL": "m", "BRINE_API_KEY": ""})


def test_api_key_not_in_repr():
    s = LLMSettings(url="u", model="m", api_key="super-secret")
    assert "super-secret" not in repr(s)


# --- live endpoint --------------------------------------------------------


@live
def test_live_chat_returns_text_and_token_usage():
    with LLMClient(LLMSettings.from_env()) as client:
        result = client.chat(
            [{"role": "user", "content": "Reply with the single word OK."}],
            max_tokens=256,
            thinking=False,
        )
    assert result.text.strip()
    assert result.prompt_tokens is not None and result.prompt_tokens > 0
    assert result.completion_tokens is not None and result.completion_tokens > 0


@live
def test_live_wrong_api_key_is_rejected_without_retry():
    settings = LLMSettings.from_env()
    bad = LLMSettings(url=settings.url, model=settings.model, api_key="invalid-key-for-test")
    sleeps: list[float] = []
    with LLMClient(bad, retries=2, sleep=sleeps.append) as client:
        with pytest.raises(httpx.HTTPStatusError) as exc:
            client.chat([{"role": "user", "content": "hi"}], max_tokens=8, thinking=False)
    assert exc.value.response.status_code in (401, 403)
    assert sleeps == []  # 4xx must not trigger retries


@live
def test_live_unknown_model_fails_fast():
    settings = LLMSettings.from_env()
    bad = LLMSettings(url=settings.url, model="no-such-model-xyz", api_key=settings.api_key)
    with LLMClient(bad, retries=0) as client:
        with pytest.raises(httpx.HTTPStatusError) as exc:
            client.chat([{"role": "user", "content": "hi"}], max_tokens=8, thinking=False)
    assert 400 <= exc.value.response.status_code < 500
