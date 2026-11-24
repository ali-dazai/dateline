"""Model fallback, timeouts, deadlines and usage logging in dateline.llm (no network)."""
import csv
import json
from types import SimpleNamespace

import litellm
import pytest

from dateline import config, llm


class ServiceUnavailableError(Exception):
    """Same class name as litellm's, so llm treats it as retryable."""


def reply(text, usage=None):
    resp = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])
    if usage is not None:
        resp.usage = SimpleNamespace(**usage)
    return resp


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache" / "llm")
    monkeypatch.setattr(config, "MIN_INTERVAL_S", 0.0)
    monkeypatch.setattr(config, "ANSWER_MODEL", "primary/model")
    monkeypatch.setattr(config, "FALLBACK_MODEL", "fallback/model")
    monkeypatch.setattr(llm, "_sleep", lambda s: None)


def stub(monkeypatch, handler):
    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        return handler(kwargs)

    monkeypatch.setattr(litellm, "completion", fake)
    return calls


def primary_down(kwargs):
    if kwargs["model"] == "primary/model":
        raise ServiceUnavailableError("503 unavailable")
    return reply("ok")


def test_fallback_used_when_primary_keeps_failing(monkeypatch):
    calls = stub(monkeypatch, primary_down)
    assert llm.complete("q", use_cache=False) == "ok"
    models = [c["model"] for c in calls]
    assert models.count("primary/model") == 8
    assert models[-1] == "fallback/model"


def test_no_fallback_configured_reraises(monkeypatch):
    monkeypatch.setattr(config, "FALLBACK_MODEL", None)
    stub(monkeypatch, primary_down)
    with pytest.raises(ServiceUnavailableError):
        llm.complete("q", use_cache=False)


def test_fallback_equal_to_primary_is_not_retried_again(monkeypatch):
    monkeypatch.setattr(config, "FALLBACK_MODEL", "primary/model")
    calls = stub(monkeypatch, primary_down)
    with pytest.raises(ServiceUnavailableError):
        llm.complete("q", use_cache=False)
    assert len(calls) == 8


def test_timeout_passed_to_litellm(monkeypatch):
    calls = stub(monkeypatch, lambda k: reply("hi"))
    llm.complete("q", use_cache=False)
    assert calls[0]["timeout"] == config.CALL_TIMEOUT_S == 60


def test_max_tokens_passed_only_when_given(monkeypatch):
    calls = stub(monkeypatch, lambda k: reply("hi"))
    llm.complete("q", use_cache=False)
    llm.complete("q", use_cache=False, max_tokens=200)
    assert "max_tokens" not in calls[0]
    assert calls[1]["max_tokens"] == 200


def test_attempts_limit_then_one_fallback_try(monkeypatch):
    calls = stub(monkeypatch, primary_down)
    assert llm.complete("q", use_cache=False, attempts=2, deadline_s=30) == "ok"
    assert [c["model"] for c in calls] == ["primary/model"] * 2 + ["fallback/model"]


def test_attempts_limit_raises_when_fallback_also_fails(monkeypatch):
    calls = stub(monkeypatch, lambda k: (_ for _ in ()).throw(ServiceUnavailableError("503")))
    with pytest.raises(ServiceUnavailableError):
        llm.complete("q", use_cache=False, attempts=2, deadline_s=30)
    assert [c["model"] for c in calls] == ["primary/model"] * 2 + ["fallback/model"]


def test_deadline_stops_retries_and_caps_call_timeout(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(llm, "_now", lambda: now[0])
    monkeypatch.setattr(llm, "_sleep", lambda s: now.__setitem__(0, now[0] + s))

    def slow_fail(kwargs):
        now[0] += 10  # each failing call burns 10 s
        return primary_down(kwargs)

    calls = stub(monkeypatch, slow_fail)
    with pytest.raises(ServiceUnavailableError):
        llm.complete("q", use_cache=False, deadline_s=25, attempts=5)
    assert all(c["timeout"] <= 25 for c in calls)
    assert now[0] - 100.0 <= 25 + 10  # never starts a call past the deadline
    assert len([c for c in calls if c["model"] == "primary/model"]) < 5


def test_fallback_result_cached_under_primary_key(monkeypatch):
    calls = stub(monkeypatch, primary_down)
    assert llm.complete("q") == "ok"
    n = len(calls)
    assert llm.complete("q") == "ok"
    assert len(calls) == n
    stored = list(config.CACHE_DIR.glob("*.json"))
    assert len(stored) == 1
    assert json.loads(stored[0].read_text(encoding="utf-8"))["request"]["model"] == "primary/model"


def test_max_tokens_in_cache_key_only_when_given(monkeypatch):
    stub(monkeypatch, lambda k: reply("hi"))
    llm.complete("q")
    old_key = next(config.CACHE_DIR.glob("*.json")).name
    expected = llm._cache_key({"model": "primary/model", "messages": [{"role": "user", "content": "q"}],
                               "temperature": 0.0, "json": False})
    assert old_key == f"{expected}.json"
    llm.complete("q", max_tokens=50)
    assert len(list(config.CACHE_DIR.glob("*.json"))) == 2


def test_complete_json_forwards_new_kwargs(monkeypatch):
    seen = {}

    def fake_complete(prompt, **kwargs):
        seen.update(kwargs)
        return '{"a": 1}'

    monkeypatch.setattr(llm, "complete", fake_complete)
    assert llm.complete_json("q", max_tokens=9, deadline_s=5, attempts=1) == {"a": 1}
    assert seen["max_tokens"] == 9 and seen["deadline_s"] == 5 and seen["attempts"] == 1 and seen["json_mode"]


def test_usage_logged_per_real_call_not_cached(monkeypatch):
    stub(monkeypatch, lambda k: reply("hi", {"prompt_tokens": 12, "completion_tokens": 3}))
    llm.complete("q")
    llm.complete("q")  # cached: no new row
    rows = list(csv.DictReader((config.CACHE_DIR.parent / "usage.csv").open(encoding="utf-8")))
    assert len(rows) == 1
    assert list(rows[0]) == ["time", "model", "prompt_tokens", "completion_tokens", "seconds"]
    assert rows[0]["model"] == "primary/model"
    assert (rows[0]["prompt_tokens"], rows[0]["completion_tokens"]) == ("12", "3")


def test_usage_logging_failure_never_breaks_the_call(monkeypatch):
    stub(monkeypatch, lambda k: reply("hi"))
    monkeypatch.setattr(config, "CACHE_DIR", config.CACHE_DIR.parent / "llm")
    (config.CACHE_DIR.parent).mkdir(parents=True, exist_ok=True)
    (config.CACHE_DIR.parent / "usage.csv").mkdir()  # a directory: opening it for append fails
    assert llm.complete("q", use_cache=False) == "hi"


def test_openrouter_calls_turn_thinking_off(monkeypatch):
    monkeypatch.setattr(config, "ANSWER_MODEL", "openrouter/anthropic/claude-haiku-5.5")
    monkeypatch.setattr(config, "REASONING_EFFORT", "none")
    calls = stub(monkeypatch, lambda kwargs: reply("ok"))
    llm.complete("q", use_cache=False)
    llm.complete("q", model="fallback/model", use_cache=False)
    assert calls[0]["reasoning_effort"] == "none"
    assert "reasoning_effort" not in calls[1]


def test_reasoning_setting_is_part_of_the_cache_key(monkeypatch):
    monkeypatch.setattr(config, "ANSWER_MODEL", "openrouter/anthropic/claude-haiku-5.5")
    replies = iter(["first", "second"])
    stub(monkeypatch, lambda kwargs: reply(next(replies)))
    monkeypatch.setattr(config, "REASONING_EFFORT", "")
    assert llm.complete("q") == "first"
    monkeypatch.setattr(config, "REASONING_EFFORT", "none")
    assert llm.complete("q") == "second"


def test_empty_reply_is_not_cached(monkeypatch):
    replies = iter(["", "ok"])
    calls = stub(monkeypatch, lambda kwargs: reply(next(replies)))
    assert llm.complete("q") == ""
    assert llm.complete("q") == "ok"
    assert llm.complete("q") == "ok"
    assert len(calls) == 2


def test_cache_written_atomically(monkeypatch):
    stub(monkeypatch, lambda kwargs: reply("first"))
    assert llm.complete("hello") == "first"
    files = list(config.CACHE_DIR.iterdir())
    assert len(files) == 1 and files[0].suffix == ".json"
    assert json.loads(files[0].read_text(encoding="utf-8"))["response"] == "first"


def test_failed_cache_write_leaves_no_partial_file(monkeypatch):
    stub(monkeypatch, lambda kwargs: reply("second"))

    def broken_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(llm.os, "replace", broken_replace)
    assert llm.complete("hello again") == "second"
    assert list(config.CACHE_DIR.iterdir()) == []
