"""Stage tracing: payloads are clipped, models are shown as JSON, debug mode is opt-in."""

import logging

from causa_nostra.pipeline.trace import MAX_PAYLOAD_CHARS, clip, debug_requested, payload
from causa_nostra.pipeline.types import QuestionFrame


def test_clip_keeps_short_text_and_marks_cut_text():
    assert clip("abc", limit=5) == "abc"
    cut = clip("x" * 12, limit=5)
    assert cut.startswith("xxxxx")
    assert cut.endswith("[+7 chars]")


def test_payload_shows_models_as_json():
    frame = QuestionFrame(question_id="T-1", intervention="burning", outcome="grass cover")
    text = payload(frame)
    assert '"intervention":"burning"' in text.replace(" ", "")


def test_payload_is_bounded_for_long_values():
    assert len(payload("y" * (MAX_PAYLOAD_CHARS * 2))) < MAX_PAYLOAD_CHARS + 100


def test_debug_is_off_by_default(monkeypatch):
    monkeypatch.delenv("CAUSA_DEBUG", raising=False)
    assert debug_requested(False) is False
    assert debug_requested(True) is True


def test_env_var_turns_debug_on(monkeypatch):
    monkeypatch.setenv("CAUSA_DEBUG", "1")
    assert debug_requested(False) is True


def test_stage_debug_is_skipped_when_debug_is_off(caplog):
    from causa_nostra.pipeline.trace import TRACE, stage_debug

    caplog.set_level(logging.INFO, logger=TRACE.name)
    stage_debug("T-1", "retrieve", "output", ["secret"])
    assert "secret" not in caplog.text
