from datetime import date

import pytest
from fastapi.testclient import TestClient

import api.server as server
from api.server import app
from dateline import config
from eval import baseline

client = TestClient(app)


def test_health():
    assert client.get("/api/health").json()["ok"] is True


def test_answer_follows_contract():
    r = client.post("/api/answer", json={"question": "What happened to the Titanic?", "cutoff": "1912-04-15", "guard": True})
    assert r.status_code == 200
    body = r.json()
    assert body["problems"] == [] and body["sentences"] and body["removed"]


def test_cutoff_outside_corpus_is_rejected():
    r = client.post("/api/answer", json={"question": "What happened?", "cutoff": "1913-01-01"})
    assert r.status_code == 422


def test_events_listed():
    ids = [e["id"] for e in client.get("/api/events").json()]
    assert ids == ["titanic-1912", "lehman-2008", "titan-2023"]


def test_range_has_counts():
    body = client.get("/api/range", params={"event": "lehman-2008"}).json()
    assert body["granularity"] == "day" and body["counts"]
    assert all(c["bucket"][:10] >= "2008-09-10" for c in body["counts"])


def test_lehman_rescue_then_bankruptcy():
    ask = lambda c: client.post("/api/answer", json={"question": "Will Lehman be rescued?", "cutoff": c,
                                                     "event": "lehman-2008"}).json()
    friday, monday = ask("2008-09-12"), ask("2008-09-15")
    assert friday["problems"] == [] and monday["problems"] == []
    assert not any("bankruptcy" in s["text"] for s in friday["sentences"])
    assert any("bankruptcy" in s["text"] for s in monday["sentences"])
    assert all(0 <= s["group"] < len(monday["groups"]) for s in monday["sentences"])


def test_titan_minute_cutoff():
    ask = lambda c: client.post("/api/answer", json={"question": "Is the crew alive?", "cutoff": c,
                                                     "event": "titan-2023", "guard": True}).json()
    before, after = ask("2023-06-22T15:54:00Z"), ask("2023-06-22T15:56:00Z")
    assert not any("debris" in s["text"] for s in before["sentences"]) and before["removed"]
    assert any("debris" in s["text"] for s in after["sentences"]) and not after["removed"]
    assert before["problems"] == [] and after["problems"] == []


def test_unknown_event_404():
    r = client.post("/api/answer", json={"question": "What?", "cutoff": "1912-04-15", "event": "nope"})
    assert r.status_code == 404


# --- the engine call: interactive limits and failures -----------------------------------------------


def _stub_engine(monkeypatch, fn):
    monkeypatch.setattr(server, "get_answer", lambda: fn)


def test_answer_passes_interactive_when_the_engine_accepts_it(monkeypatch):
    seen = {}

    def engine(question, cutoff, guard=True, event="titanic-1912", *, interactive=False):
        seen["interactive"] = interactive
        return {"sentences": [], "removed": [], "groups": [], "retrieved": []}

    _stub_engine(monkeypatch, engine)
    r = client.post("/api/answer", json={"question": "What happened?", "cutoff": "1912-04-15"})
    assert r.status_code == 200 and seen == {"interactive": True}


def test_answer_leaves_out_interactive_for_engines_without_it(monkeypatch):
    def engine(question, cutoff, guard=True, event="titanic-1912"):
        return {"sentences": [], "removed": [], "groups": [], "retrieved": []}

    _stub_engine(monkeypatch, engine)
    r = client.post("/api/answer", json={"question": "What happened?", "cutoff": "1912-04-15"})
    assert r.status_code == 200 and r.json()["problems"] == []


def test_engine_exception_becomes_a_notice(monkeypatch):
    def engine(question, cutoff, guard=True, event="titanic-1912", *, interactive=False):
        raise RuntimeError("model went away")

    _stub_engine(monkeypatch, engine)
    r = client.post("/api/answer", json={"question": "What happened?", "cutoff": "1912-04-15"})
    assert r.status_code == 200
    assert r.json() == {"sentences": [], "removed": [], "groups": [], "retrieved": [],
                        "notice": "The model could not produce an answer. Try again.", "problems": []}


def test_engine_without_event_support_still_501(monkeypatch):
    def engine(question, cutoff, guard=True, event="titanic-1912"):
        raise NotImplementedError("this engine only supports titanic-1912")

    _stub_engine(monkeypatch, engine)
    r = client.post("/api/answer", json={"question": "Will Lehman be rescued?", "cutoff": "2008-09-12",
                                         "event": "lehman-2008"})
    assert r.status_code == 501


def test_bad_input_still_422_with_a_stubbed_engine(monkeypatch):
    _stub_engine(monkeypatch, lambda *a, **k: pytest.fail("engine called for bad input"))
    assert client.post("/api/answer", json={"question": "What?", "cutoff": "April 15"}).status_code == 422
    assert client.post("/api/answer", json={"question": "Wh", "cutoff": "1912-04-15"}).status_code == 422


# --- the plain model: told the date, no search ------------------------------------------------------

# The prompt-only baseline's system prompt for April 15, 1912, as eval/baseline.py has always sent it.
DAY_SYSTEM_1912 = (
    "You are a newspaper reader living in the end of Monday, April 15, 1912. You have read every newspaper "
    "printed up to and including that day, and you know nothing that happened or was reported later. "
    "Answer the question using only what was known at that time. If newspapers disagreed, say so. "
    "When you rely on a specific newspaper, quote its words in double quotes and name the paper."
)


@pytest.fixture
def calls(monkeypatch):
    seen: list[dict] = []

    def complete(prompt, **kwargs):
        seen.append({"prompt": prompt, **kwargs})
        return "  The papers said all were saved.  "

    monkeypatch.setattr(baseline.llm, "complete", complete)
    return seen


def test_plain_answer_uses_baseline(monkeypatch):
    seen = {}

    def fake(question, cutoff, model=None, *, when=None, **limits):
        seen.update(question=question, cutoff=cutoff, when=when, limits=limits)
        return "All saved."

    monkeypatch.setattr(server.baseline, "answer", fake)
    r = client.post("/api/plain", json={"question": "What happened to the Titanic?", "cutoff": "1912-04-15",
                                        "event": "titanic-1912"})
    assert r.status_code == 200
    assert r.json() == {"answer": "All saved.", "moment": "the end of Monday, April 15, 1912"}
    assert seen["question"] == "About the Titanic (April 1912): What happened to the Titanic?"
    assert str(seen["cutoff"]) == "1912-04-15"
    assert seen["limits"] == {"attempts": 2, "timeout_s": 20.0, "deadline_s": 30.0}
    assert client.post("/api/plain", json={"question": "What happened?", "cutoff": "15 April"}).status_code == 422
    assert client.post("/api/plain", json={"question": "What happened?", "cutoff": "1913-01-01"}).status_code == 422
    assert client.post("/api/plain", json={"question": "Wh", "cutoff": "1912-04-15"}).status_code == 422
    assert client.post("/api/plain", json={"question": "What happened?", "cutoff": "1912-04-15",
                                           "event": "nope"}).status_code == 404


def test_plain_day_prompt_is_byte_identical_to_the_baseline(calls):
    r = client.post("/api/plain", json={"question": "What happened to the Titanic?", "cutoff": "1912-04-15"})
    assert r.json() == {"answer": "The papers said all were saved.", "moment": "the end of Monday, April 15, 1912"}
    baseline.answer("What happened to the Titanic?", date(1912, 4, 15))  # the evaluation's call
    api_call, eval_call = calls
    for call in calls:
        assert call["system"] == DAY_SYSTEM_1912
        assert call["model"] == config.ANSWER_MODEL
    assert eval_call["prompt"] == "What happened to the Titanic?"  # the evaluation's prompt is unchanged
    # The page names the story, as Dateline's search does (story terms); the system prompt is the same.
    assert api_call["prompt"] == "About the Titanic (April 1912): What happened to the Titanic?"
    assert {k: api_call[k] for k in ("attempts", "timeout_s", "deadline_s")} == \
        {"attempts": 2, "timeout_s": 20.0, "deadline_s": 30.0}
    assert set(eval_call) == {"prompt", "system", "model"}  # batch defaults: no limits passed


def test_plain_minute_story_is_told_the_minute(calls):
    r = client.post("/api/plain", json={"question": "Is the crew alive?", "cutoff": "2023-06-21T14:00:00-04:00",
                                        "event": "titan-2023"})
    assert r.status_code == 200 and r.json()["moment"] == "2:00 p.m. EDT, Wednesday, June 21, 2023"
    assert "living in 2:00 p.m. EDT, Wednesday, June 21, 2023. You have read" in calls[0]["system"]


def test_plain_model_failure_is_a_notice(monkeypatch):
    def broken(prompt, **kwargs):
        raise TimeoutError("no reply")

    monkeypatch.setattr(baseline.llm, "complete", broken)
    r = client.post("/api/plain", json={"question": "What happened?", "cutoff": "1912-04-15"})
    assert r.status_code == 200
    assert r.json() == {"answer": "", "moment": "the end of Monday, April 15, 1912",
                        "notice": "The plain model could not answer. Try again."}


def test_plain_empty_reply_is_a_notice(monkeypatch):
    monkeypatch.setattr(baseline.llm, "complete", lambda prompt, **kwargs: "  ")
    r = client.post("/api/plain", json={"question": "What happened?", "cutoff": "1912-04-15"})
    assert r.json()["answer"] == "" and r.json()["notice"] == "The plain model could not answer. Try again."


# --- warm start: the model library loads in the background ------------------------------------------

def test_startup_loads_the_model_library_without_blocking(monkeypatch):
    import threading
    import time

    release, started = threading.Event(), threading.Event()
    threads = []

    def slow_preload():
        threads.append(threading.current_thread())
        started.set()
        release.wait(10)  # stands in for the ~10 s LiteLLM import

    monkeypatch.setattr(server.llm, "preload", slow_preload)
    begun = time.perf_counter()
    try:
        with TestClient(app) as c:
            assert c.get("/api/health").json()["ok"] is True
            assert time.perf_counter() - begun < 2
            assert started.wait(2)
            assert threads and threads[0] is not threading.main_thread() and threads[0].daemon
    finally:
        release.set()


def test_preload_failure_is_harmless(monkeypatch):
    def broken():
        raise ImportError("no litellm")

    monkeypatch.setattr(server.llm, "preload", broken)
    with TestClient(app) as c:
        assert c.get("/api/health").status_code == 200


def test_startup_runs_stored_presets_through_the_engine(monkeypatch):
    """The quote check's first pass over an answer is slow (~0.5 s); doing it at startup keeps every
    preset instant. Only answers already stored are run, so no model is called in the background."""
    import threading
    from datetime import date, datetime, timezone

    from pipeline import warm

    done = threading.Event()
    seen = []

    def engine(question, cutoff, guard=True, event="titanic-1912", *, interactive=False):
        seen.append((question, cutoff, guard, event, interactive, threading.current_thread().name))
        return {"sentences": [], "removed": [], "groups": [], "retrieved": []}

    items = [warm.Item("titanic-1912", "Stored?", "1912-04-15", "preset"),
             warm.Item("titanic-1912", "Not stored", "1912-04-16", "preset"),
             warm.Item("titan-2023", "Minute?", "2023-06-21T08:00:00-04:00", "preset")]
    monkeypatch.setattr(server.warm, "items", lambda: items)
    monkeypatch.setattr(server.pipeline_engine, "stored", lambda question, cutoff, event: question != "Not stored")
    monkeypatch.setattr(server, "get_answer", lambda: engine)
    monkeypatch.setattr(server.llm, "preload", done.set)
    with TestClient(app) as c:
        assert c.get("/api/health").status_code == 200
        assert done.wait(5)
    assert [s[0] for s in seen] == ["Stored?", "Minute?"]
    assert seen[0][1] == date(1912, 4, 15) and seen[1][1] == datetime(2023, 6, 21, 12, 0, tzinfo=timezone.utc)
    assert all(s[2] is True and s[4] is True and s[5] != threading.main_thread().name for s in seen)


def test_startup_skips_presets_for_an_engine_without_interactive_limits(monkeypatch):
    import threading

    from pipeline import warm

    done = threading.Event()
    calls = []
    monkeypatch.setattr(server.warm, "items", lambda: [warm.Item("titanic-1912", "Q?", "1912-04-15", "preset")])
    monkeypatch.setattr(server.pipeline_engine, "stored", lambda question, cutoff, event: True)
    monkeypatch.setattr(server, "get_answer", lambda: (lambda *a, **k: calls.append(a)))
    monkeypatch.setattr(server.llm, "preload", done.set)
    with TestClient(app) as c:
        c.get("/api/health")
        assert done.wait(5)
    assert calls == []


def test_a_broken_preset_list_still_loads_the_model_library(monkeypatch):
    import threading

    done = threading.Event()

    def broken_items():
        raise ValueError("presets.json is not valid JSON")

    monkeypatch.setattr(server.warm, "items", broken_items)
    monkeypatch.setattr(server, "get_answer", lambda: pipeline_like_engine)
    monkeypatch.setattr(server.llm, "preload", done.set)
    with TestClient(app) as c:
        assert c.get("/api/health").status_code == 200
        assert done.wait(5)


def pipeline_like_engine(question, cutoff, guard=True, event="titanic-1912", *, interactive=False):
    return {"sentences": [], "removed": [], "groups": [], "retrieved": []}
