"""Cache warm-up: every preset and each story's default answered and stored; --check makes no model call."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from dateline import config
from pipeline import warm

ROOT = Path(__file__).resolve().parents[2]

DAY_Q = "Were the Titanic passengers safe?"
DAY_DEFAULT = "What did the papers say about the Titanic?"
MIN_Q = "Were banging sounds heard in the search for Titan?"
GOOD_DAY = {"text": "The Washington Times reported that the passengers are safe.",
            "quote": "he was assured the Titanic would reach port under her own steam, and that the passengers are safe",
            "article_id": "day-06", "claim": "the passengers are safe"}
SECOND_DAY = {"text": "The Daily Missoulian reported that her people are in no danger.",
              "quote": "The steamer is believed to be unhurt and her people are in no danger",
              "article_id": "day-04", "claim": "her people are in no danger"}
GOOD_MIN = {"text": "The Guardian reported that search aircraft heard banging noises at 30-minute intervals.",
            "quote": "Search aircraft heard banging noises at 30-minute intervals, officials said",
            "article_id": "min-04", "claim": "banging noises heard"}


class FakeLiteLLM(types.ModuleType):
    """Stands in for the litellm module: routes each request by its first line and counts the calls."""

    def __init__(self, compose: str | None = None) -> None:
        super().__init__("litellm")
        self.calls: list[str] = []
        self.compose = compose

    def completion(self, **kwargs):
        system = next((m["content"] for m in kwargs["messages"] if m["role"] == "system"), "")
        prompt = kwargs["messages"][-1]["content"]
        if system.startswith("You are a newspaper reader"):
            task, text = "plain", "The papers said everyone was saved."
        else:
            task = prompt.lstrip().splitlines()[0].split(":", 1)[1].strip()
            text = {
                "terms": json.dumps({"terms": ["rescued", "saved"]}),
                "rerank": json.dumps({"ids": ["day-06", "day-04", "min-04", "min-05"]}),
                "classify": json.dumps({"assign": {}}),
                "compose": self.compose or json.dumps({
                    "sentences": [GOOD_MIN] if "Titan?" in prompt or "banging" in prompt else [GOOD_DAY, SECOND_DAY],
                    "claims": {}, "figures": []}),
            }[task]
        self.calls.append(task)
        message = types.SimpleNamespace(content=text)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)], usage=None)


@pytest.fixture
def fake_litellm(monkeypatch):
    fake = FakeLiteLLM()
    monkeypatch.setitem(sys.modules, "litellm", fake)
    monkeypatch.setattr(config, "MIN_INTERVAL_S", 0.0)
    monkeypatch.setattr(config, "FALLBACK_MODEL", "")
    return fake


@pytest.fixture
def files(tmp_path, day_event, minute_event):
    presets = tmp_path / "presets.json"
    presets.write_text(json.dumps({
        day_event.event: [{"question": DAY_DEFAULT, "cutoff": "1912-04-15"},
                          {"question": DAY_Q, "cutoff": "1912-04-15", "label": "Safe?"}],
        minute_event.event: [{"question": MIN_Q, "cutoff": "2023-06-21T06:00:00-04:00"}],
    }), encoding="utf-8")
    stories = tmp_path / "stories.ts"
    stories.write_text(
        "export const STORY_META: Record<string, Meta> = {\n"
        f"  '{day_event.event}': {{\n    title: 'Day',\n    question: '{DAY_DEFAULT}',\n"
        "    default_cutoff: '1912-04-15',\n  },\n"
        f"  '{minute_event.event}': {{\n    title: 'Minute',\n    question: 'Was the sub\\'s crew found?',\n"
        "    start: '2023-06-18T00:00:00-04:00',\n    default_cutoff: '2023-06-21T06:00:00-04:00',\n  },\n}\n",
        encoding="utf-8")
    return presets, stories


def test_defaults_read_from_the_web_app_stories_file():
    found = {(d.event, d.question, d.cutoff) for d in warm.defaults(ROOT / "web" / "src" / "stories.ts")}
    assert ("titanic-1912", "What happened to the Titanic’s passengers?", "1912-04-15") in found
    assert ("lehman-2008", "Will Lehman Brothers be rescued?", "2008-09-12") in found
    assert ("titan-2023", "What happened to the Titan submersible?", "2023-06-21T14:00:00-04:00") in found
    assert len(found) == 3


def test_items_put_each_default_first_and_skip_duplicates(files, day_event, minute_event):
    items = warm.items(*files)
    assert [(i.event, i.question, i.kind) for i in items] == [
        (day_event.event, DAY_DEFAULT, "default"),  # also a preset: listed once
        (day_event.event, DAY_Q, "preset"),
        (minute_event.event, "Was the sub's crew found?", "default"),
        (minute_event.event, MIN_Q, "preset"),
    ]


def test_check_before_warming_lists_every_item_and_calls_no_model(files, fake_litellm, capsys):
    problems = warm.run(warm.items(*files), check=True)
    assert len(problems) == 4 and all("would call the model" in p for p in problems)
    assert fake_litellm.calls == []  # the guard stood in for the model
    assert sys.modules["litellm"] is fake_litellm and "completion" not in vars(fake_litellm)  # restored
    assert warm.main(["--check", "--presets", str(files[0]), "--stories", str(files[1])]) == 1
    assert "would call the model" in capsys.readouterr().out


def test_warm_answers_everything_then_check_passes(files, fake_litellm, capsys):
    items = warm.items(*files)
    assert warm.run(items) == []
    first = capsys.readouterr().out
    assert "plain: new" in first and "guard on: new" in first
    assert "guard off: hit" in first  # guard on and off share one stored model answer
    assert fake_litellm.calls.count("plain") == 4
    assert {"terms", "rerank", "compose"} <= set(fake_litellm.calls)
    assert all(warm.answer_stored(i) for i in items)

    calls = len(fake_litellm.calls)
    assert warm.run(items) == []
    again = capsys.readouterr().out
    assert len(fake_litellm.calls) == calls  # all from the caches
    assert again.count("guard on: hit | guard off: hit | plain: hit") == 4

    assert warm.run(items, check=True) == []
    assert warm.main(["--check", "--presets", str(files[0]), "--stories", str(files[1])]) == 0
    assert len(fake_litellm.calls) == calls


def test_check_does_not_import_the_model_library(files, fake_litellm, monkeypatch):
    warm.run(warm.items(*files))
    monkeypatch.delitem(sys.modules, "litellm")
    assert warm.run(warm.items(*files), check=True) == []
    assert "litellm" not in sys.modules  # the stand-in guard module is taken out again


def test_check_blocks_a_real_import_too(files, monkeypatch):
    """With the library already imported, --check replaces its completion and puts it back."""
    real = FakeLiteLLM()
    monkeypatch.setitem(sys.modules, "litellm", real)
    monkeypatch.setattr(config, "MIN_INTERVAL_S", 0.0)
    monkeypatch.setattr(config, "FALLBACK_MODEL", "")
    original = real.completion
    problems = warm.run(warm.items(*files)[:1], check=True)
    assert problems and real.calls == []
    assert real.completion == original


def test_warm_reports_an_answer_that_could_not_be_stored(files, monkeypatch):
    broken = FakeLiteLLM(compose="not json at all")
    monkeypatch.setitem(sys.modules, "litellm", broken)
    monkeypatch.setattr(config, "MIN_INTERVAL_S", 0.0)
    monkeypatch.setattr(config, "FALLBACK_MODEL", "")
    problems = warm.run(warm.items(*files)[:1])
    assert any("engine answer not stored" in p for p in problems)


# --- --check also fails items that break without reaching the model ---------------------------------

def test_check_fails_an_item_that_raises_without_a_model_call(files, fake_litellm, monkeypatch):
    from pipeline import engine

    todo = warm.items(*files)
    assert warm.run(todo) == []

    def broken(*_args, **_kwargs):
        raise RuntimeError("bug in the checks")

    monkeypatch.setattr(engine, "_finish", broken)
    problems = warm.run(todo, check=True)
    assert len(problems) == len(todo) and all("RuntimeError" in p for p in problems)
    assert fake_litellm.calls.count("plain") == 4  # still no new model call


def test_check_fails_an_unreadable_cached_reply(files, fake_litellm):
    todo = warm.items(*files)
    assert warm.run(todo) == []
    for path in config.CACHE_DIR.glob("*.json"):
        path.write_text("{not json", encoding="utf-8")
    problems = warm.run(todo, check=True)
    assert problems and all("plain" in p and "JSONDecodeError" in p for p in problems)


def test_check_fails_an_item_answered_with_a_notice(files, fake_litellm, monkeypatch, day_event):
    from pipeline import index

    todo = warm.items(*files)
    assert warm.run(todo) == []
    monkeypatch.setattr(index, "index_path", lambda event: day_event.index.with_name("gone.sqlite"))
    problems = warm.run(todo, check=True)
    assert len(problems) == len(todo)
    assert all("notice: The search index for this story has not been built yet." in p for p in problems)
    assert warm.main(["--check", "--presets", str(files[0]), "--stories", str(files[1])]) == 1
