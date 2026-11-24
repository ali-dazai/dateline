"""The engine: the whole answer flow, the answer cache, notices and interactive limits."""
from __future__ import annotations

import json
import threading
import time
from datetime import date, datetime, timezone

import pytest

from dateline import config, events, llm
from dateline.contract import validate_result
from pipeline import engine, index
from pipeline import search as search_mod
from pipeline.query import question_terms

DAY_Q = "Were the Titanic passengers safe?"
DAY_CUTOFF = date(1912, 4, 15)
GOOD_DAY = {"text": "The Washington Times reported that the passengers are safe.",
            "quote": "he was assured the Titanic would reach port under her own steam, and that the passengers are safe",
            "article_id": "day-06", "claim": "the passengers are safe"}
HINDSIGHT_DAY = {"text": "The Titanic sank and 1,500 lost their lives.",
                 "quote": "he was assured the Titanic would reach port under her own steam, and that the passengers are safe",
                 "article_id": "day-06", "claim": "the passengers are safe"}
SECOND_DAY = {"text": "The Daily Missoulian reported that her people are in no danger.",
              "quote": "The steamer is believed to be unhurt and her people are in no danger",
              "article_id": "day-04", "claim": "her people are in no danger"}

MIN_Q = "Were banging sounds heard in the search for Titan?"
MIN_CUTOFF = datetime(2023, 6, 21, 10, 0, tzinfo=timezone.utc)
GOOD_MIN = {"text": "The Guardian reported that search aircraft heard banging noises at 30-minute intervals.",
            "quote": "Search aircraft heard banging noises at 30-minute intervals, officials said",
            "article_id": "min-04", "claim": "banging noises heard"}


@pytest.fixture(autouse=True)
def _answer_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache" / "llm")


def _compose(*sentences, claims=None, figures=()):
    claims = claims if claims is not None else {s["article_id"]: s["claim"] for s in sentences}
    return json.dumps({"sentences": list(sentences), "claims": claims, "figures": list(figures)})


def _script(stub, compose, ids=("day-06", "day-04", "day-02"), classify=None):
    """Stateless replies (safe to share between threads), compose replies taken in order."""
    stub.replies["terms"] = lambda prompt: json.dumps({"terms": ["rescued", "saved"]})
    stub.replies["rerank"] = lambda prompt: json.dumps({"ids": list(ids)})
    stub.replies["compose"] = compose if callable(compose) else list(compose)
    stub.replies["classify"] = classify or (lambda prompt: json.dumps({"assign": {}}))


def _tasks(stub):
    return [task for task, _ in stub.calls]


def _valid(result, cutoff):
    assert validate_result(result, cutoff) == []
    return result


def _answer(question, cutoff, **kwargs):
    """engine.answer, with the result checked against the contract."""
    return _valid(engine.answer(question, cutoff, **kwargs), cutoff)


def _stored():
    return sorted((config.CACHE_DIR.parent / "answers").glob("*.json"))


# --- end to end ------------------------------------------------------------------------------------

def test_day_story_end_to_end(day_event, stub_llm):
    _script(stub_llm, [_compose(GOOD_DAY, SECOND_DAY)])
    result = _valid(engine.answer(DAY_Q, DAY_CUTOFF, event=day_event.event), DAY_CUTOFF)
    assert [s["text"] for s in result["sentences"]] == [GOOD_DAY["text"], SECOND_DAY["text"]]
    assert result["removed"] == []
    assert "notice" not in result
    first = result["sentences"][0]
    assert first["paper"] == "The Washington Times" and first["date"] == "1912-04-15"
    assert first["url"].startswith("https://")
    assert 0 <= first["group"] < len(result["groups"])
    assert {g["claim"] for g in result["groups"]} >= {"the passengers are safe", "her people are in no danger"}
    assert all(i.startswith("day-") for i in result["retrieved"])
    assert "day-09" not in result["retrieved"]  # published April 16
    assert _tasks(stub_llm) == ["terms", "rerank", "compose", "classify"]


def test_minute_story_end_to_end(minute_event, stub_llm):
    _script(stub_llm, [_compose(GOOD_MIN)], ids=("min-04", "min-05"))
    result = _valid(engine.answer(MIN_Q, MIN_CUTOFF, event=minute_event.event), MIN_CUTOFF)
    assert [s["text"] for s in result["sentences"]] == [GOOD_MIN["text"]]
    assert result["sentences"][0]["published"] == "2023-06-21T05:39:00Z"
    assert "min-09" not in result["retrieved"]  # debris field, June 22
    assert "min-07" not in result["retrieved"]  # 10:01, one minute after the cutoff
    compose_prompt = [p for t, p in stub_llm.calls if t == "compose"][0]
    assert "6:00 a.m. EDT, Wednesday, June 21, 2023" in compose_prompt


def test_minute_story_allows_display_times(minute_event, stub_llm):
    timed = {"text": "At 1:39 a.m. EDT the Guardian reported that search aircraft heard banging noises.",
             "quote": GOOD_MIN["quote"], "article_id": "min-04", "claim": "banging noises heard"}
    _script(stub_llm, [_compose(timed)], ids=("min-04",))
    result = _valid(engine.answer(MIN_Q, MIN_CUTOFF, event=minute_event.event), MIN_CUTOFF)
    assert [s["text"] for s in result["sentences"]] == [timed["text"]], result["removed"]


def test_story_name_allowed_without_source(minute_event, stub_llm):
    named = {"text": "The paper reported that sonar buoys again picked up noises near the Titan search area.",
             "quote": "Sonar buoys again picked up noises near the search area", "article_id": "min-05",
             "claim": "noises heard again"}
    _script(stub_llm, [_compose(named)], ids=("min-05",))
    result = _answer("Were noises heard again?", MIN_CUTOFF, event=minute_event.event)
    assert [s["text"] for s in result["sentences"]] == [named["text"]], result["removed"]


# --- guard -----------------------------------------------------------------------------------------

def test_guard_off_keeps_removed_sentence(day_event, stub_llm):
    _script(stub_llm, [_compose(GOOD_DAY, HINDSIGHT_DAY)])
    on = _valid(engine.answer(DAY_Q, DAY_CUTOFF, guard=True, event=day_event.event), DAY_CUTOFF)
    off = _valid(engine.answer(DAY_Q, DAY_CUTOFF, guard=False, event=day_event.event), DAY_CUTOFF)
    assert [r["text"] for r in on["removed"]] == [HINDSIGHT_DAY["text"]]
    assert HINDSIGHT_DAY["text"] not in [s["text"] for s in on["sentences"]]
    assert [s["text"] for s in off["sentences"]] == [GOOD_DAY["text"], HINDSIGHT_DAY["text"]]
    assert off["removed"] == []


def test_guard_off_sets_flag(day_event, stub_llm):
    _script(stub_llm, [_compose(GOOD_DAY, HINDSIGHT_DAY)])
    off = _valid(engine.answer(DAY_Q, DAY_CUTOFF, guard=False, event=day_event.event), DAY_CUTOFF)
    good, bad = off["sentences"]
    assert "flag" not in good
    on = _answer(DAY_Q, DAY_CUTOFF, guard=True, event=day_event.event)
    assert bad["flag"] == on["removed"][0]["reason"]
    assert bad["flag"]


def test_guard_on_and_off_share_one_model_answer(day_event, stub_llm):
    _script(stub_llm, [_compose(GOOD_DAY, HINDSIGHT_DAY)])
    _answer(DAY_Q, DAY_CUTOFF, guard=True, event=day_event.event)
    _answer(DAY_Q, DAY_CUTOFF, guard=False, event=day_event.event)
    assert _tasks(stub_llm).count("compose") == 1


def test_empty_quote_always_removed(day_event, stub_llm):
    empty = {"text": "The papers said the ship was safe.", "quote": "...", "article_id": "day-06",
             "claim": "the passengers are safe"}
    _script(stub_llm, [_compose(GOOD_DAY, empty)])
    for guard in (True, False):
        result = _valid(engine.answer(DAY_Q, DAY_CUTOFF, guard=guard, event=day_event.event), DAY_CUTOFF)
        assert [s["text"] for s in result["sentences"]] == [GOOD_DAY["text"]]
        assert result["removed"] == [{"text": empty["text"], "reason": "quote shorter than 6 words"}]


def test_every_sentence_removed_gives_notice(day_event, stub_llm):
    _script(stub_llm, [_compose(HINDSIGHT_DAY)])
    result = _valid(engine.answer(DAY_Q, DAY_CUTOFF, event=day_event.event), DAY_CUTOFF)
    assert result["sentences"] == []
    assert result["notice"] == "Every sentence was removed by the quote check."
    off = _valid(engine.answer(DAY_Q, DAY_CUTOFF, guard=False, event=day_event.event), DAY_CUTOFF)
    assert len(off["sentences"]) == 1 and "notice" not in off


# --- answer cache ----------------------------------------------------------------------------------

def test_step_durations_logged(day_event, stub_llm, caplog):
    _script(stub_llm, [_compose(GOOD_DAY, SECOND_DAY)])
    with caplog.at_level("INFO", logger="pipeline"):
        _answer(DAY_Q, DAY_CUTOFF, event=day_event.event)
    steps = [r.step for r in caplog.records if r.name == "pipeline"]
    assert steps == ["terms", "search", "rerank", "passages", "compose", "classify", "checks"]
    assert all(r.seconds >= 0 for r in caplog.records if r.name == "pipeline")
    caplog.clear()
    with caplog.at_level("INFO", logger="pipeline"):
        _answer(DAY_Q, DAY_CUTOFF, event=day_event.event)  # cached: only the checks run
    assert [r.step for r in caplog.records if r.name == "pipeline"] == ["checks"]


def test_cached_answer_skips_search_and_model(day_event, stub_llm, monkeypatch):
    _script(stub_llm, [_compose(GOOD_DAY, SECOND_DAY)])
    first = _valid(engine.answer(DAY_Q, DAY_CUTOFF, event=day_event.event), DAY_CUTOFF)
    calls = len(stub_llm.calls)

    def no_search(*a, **k):
        raise AssertionError("search ran on a cached answer")

    monkeypatch.setattr(search_mod, "search", no_search)
    started = time.perf_counter()
    second = _answer(DAY_Q, DAY_CUTOFF, event=day_event.event)
    elapsed = time.perf_counter() - started
    assert len(stub_llm.calls) == calls
    assert elapsed < 0.2
    assert second == first
    files = _stored()
    assert len(files) == 1
    json.loads(files[0].read_text(encoding="utf-8"))


def test_cache_shared_by_same_normalised_question(day_event, stub_llm):
    _script(stub_llm, [_compose(GOOD_DAY)])
    _answer(DAY_Q, DAY_CUTOFF, event=day_event.event)
    _answer("  were the TITANIC passengers, safe ", DAY_CUTOFF, event=day_event.event)
    assert _tasks(stub_llm).count("compose") == 1


def test_cache_not_shared_across_moments(minute_event, stub_llm):
    # Nothing becomes available between 05:40 and 05:50, but the moment the model is told differs.
    _script(stub_llm, [_compose(GOOD_MIN), _compose(GOOD_MIN)], ids=("min-04",))
    a = datetime(2023, 6, 21, 5, 40, tzinfo=timezone.utc)
    b = datetime(2023, 6, 21, 5, 50, tzinfo=timezone.utc)
    _answer(MIN_Q, a, event=minute_event.event)
    _answer(MIN_Q, b, event=minute_event.event)
    assert _tasks(stub_llm).count("compose") == 2


def test_negated_question_not_shared(day_event, stub_llm):
    _script(stub_llm, [_compose(GOOD_DAY), _compose(GOOD_DAY)])
    _answer("Did the Titanic sink?", DAY_CUTOFF, event=day_event.event)
    _answer("Did the Titanic not sink?", DAY_CUTOFF, event=day_event.event)
    assert _tasks(stub_llm).count("compose") == 2


def test_cache_invalidated_by_version(day_event, stub_llm, monkeypatch):
    _script(stub_llm, [_compose(GOOD_DAY), _compose(GOOD_DAY)])
    _answer(DAY_Q, DAY_CUTOFF, event=day_event.event)
    monkeypatch.setattr(engine, "PIPELINE_VERSION", "changed")
    _answer(DAY_Q, DAY_CUTOFF, event=day_event.event)
    assert _tasks(stub_llm).count("compose") == 2


def test_cache_invalidated_by_model(day_event, stub_llm, monkeypatch):
    _script(stub_llm, [_compose(GOOD_DAY), _compose(GOOD_DAY)])
    _answer(DAY_Q, DAY_CUTOFF, event=day_event.event)
    monkeypatch.setattr(config, "ANSWER_MODEL", "another/model")
    _answer(DAY_Q, DAY_CUTOFF, event=day_event.event)
    assert _tasks(stub_llm).count("compose") == 2


def test_pipeline_version_hashes_pipeline_sources():
    assert len(engine.PIPELINE_VERSION) == 64
    assert engine.PIPELINE_VERSION == engine._pipeline_version()


def test_parallel_calls_from_threads(day_event, stub_llm):
    _script(stub_llm, lambda prompt: _compose(GOOD_DAY), classify=None)
    questions = [DAY_Q, DAY_Q, "Is the Titanic safe?", "Is the Titanic safe?"]
    results, errors = [None] * 4, []
    barrier = threading.Barrier(4)

    def run(i):
        try:
            barrier.wait()
            results[i] = _answer(questions[i], DAY_CUTOFF, event=day_event.event)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert errors == []
    for r in results:
        _valid(r, DAY_CUTOFF)
        assert [s["text"] for s in r["sentences"]] == [GOOD_DAY["text"]]
    assert results[0] == results[1] and results[2] == results[3]
    assert _tasks(stub_llm).count("compose") == 2  # one run per distinct question


# --- notices ---------------------------------------------------------------------------------------

def test_cutoff_before_first_article(day_event, stub_llm):
    cutoff = date(1912, 4, 13)
    result = _valid(engine.answer(DAY_Q, cutoff, event=day_event.event), cutoff)
    assert result["notice"] == "No articles had been published by this moment."
    assert result["sentences"] == [] and result["retrieved"] == []
    assert stub_llm.calls == []


@pytest.mark.parametrize("question", ["???", "Why?", "\U0001F6A2\U0001F30A", ""])
def test_unsearchable_question(day_event, stub_llm, question):
    result = _valid(engine.answer(question, DAY_CUTOFF, event=day_event.event), DAY_CUTOFF)
    assert result["notice"] == "The question has no words to search for."
    assert stub_llm.calls == []


def test_nothing_found_gives_notice(day_event, stub_llm, monkeypatch):
    monkeypatch.setitem(events.EVENTS, day_event.event, {**events.EVENTS[day_event.event], "story_terms": []})
    _script(stub_llm, [_compose(GOOD_DAY)])
    stub_llm.replies["terms"] = lambda prompt: json.dumps({"terms": []})
    result = _valid(engine.answer("Zeppelin quotations?", DAY_CUTOFF, event=day_event.event), DAY_CUTOFF)
    assert result["notice"] == "Nothing published by this moment answers this question."
    assert result["sentences"] == [] and result["retrieved"] == []
    assert "compose" not in _tasks(stub_llm)


def test_compose_failure_gives_notice(day_event, stub_llm):
    _script(stub_llm, ["not json", "still not json", _compose(GOOD_DAY)])
    result = _valid(engine.answer(DAY_Q, DAY_CUTOFF, event=day_event.event), DAY_CUTOFF)
    assert result["notice"] == "The model could not produce an answer. Try again."
    assert result["sentences"] == []
    assert result["retrieved"]
    again = _valid(engine.answer(DAY_Q, DAY_CUTOFF, event=day_event.event), DAY_CUTOFF)  # not cached
    assert [s["text"] for s in again["sentences"]] == [GOOD_DAY["text"]]


def test_index_missing_gives_notice(day_event, stub_llm):
    day_event.index.unlink()
    result = _valid(engine.answer(DAY_Q, DAY_CUTOFF, event=day_event.event), DAY_CUTOFF)
    assert result["notice"] == "The search index for this story has not been built yet."
    assert result["sentences"] == []
    assert stub_llm.calls == []
    assert not day_event.index.exists()  # never builds


# --- search wiring ---------------------------------------------------------------------------------

def test_keep_hits_merged_into_hits(day_event, stub_llm, monkeypatch):
    real = search_mod.search
    con = index.connect(day_event.event)
    keep = [h.article_id for h in real(con, question_terms(DAY_Q), events.parse_cutoff("", DAY_CUTOFF), k=3)]
    con.close()
    dropped = keep[0]

    def without_top(con, terms, cutoff, k=50, pool=None):
        hits = real(con, terms, cutoff, k=k, pool=pool)
        return hits if k == 3 else [h for h in hits if h.article_id != dropped]

    monkeypatch.setattr(search_mod, "search", without_top)
    _script(stub_llm, [_compose(GOOD_DAY)], ids=("day-06",))
    result = _valid(engine.answer(DAY_Q, DAY_CUTOFF, event=day_event.event), DAY_CUTOFF)
    assert dropped in result["retrieved"]
    con = index.connect(day_event.event)
    full = index.get_articles(con, keep)
    con.close()
    rerank_prompt = [p for t, p in stub_llm.calls if t == "rerank"][0]
    assert full[dropped]["headline"] in rerank_prompt
    compose_prompt = [p for t, p in stub_llm.calls if t == "compose"][0]
    for aid in keep:
        assert " ".join(full[aid]["text"].split()[:6]) in compose_prompt


def test_classify_gets_only_other_hits(day_event, stub_llm):
    seen = {}

    def classify(prompt):
        seen["prompt"] = prompt
        return json.dumps({"assign": {"day-05": 0}})  # a real ID is accepted as well as a label

    _script(stub_llm, [_compose(GOOD_DAY, SECOND_DAY)], ids=("day-06", "day-04"), classify=classify)
    result = _valid(engine.answer(DAY_Q, DAY_CUTOFF, event=day_event.event), DAY_CUTOFF)
    candidates = seen["prompt"].split("Candidates:", 1)[1]
    con = index.connect(day_event.event)
    full = index.get_articles(con, ["day-06", "day-04", "day-05"])
    con.close()
    opening = lambda aid: " ".join(full[aid]["text"].split()[:8])  # noqa: E731
    assert opening("day-06") not in candidates and opening("day-04") not in candidates
    assert opening("day-05") in candidates
    grouped = {a for g in result["groups"] for a in g["article_ids"]}
    assert "day-05" in grouped


def test_single_claim_skips_classify(day_event, stub_llm):
    _script(stub_llm, [_compose(GOOD_DAY)])
    _answer(DAY_Q, DAY_CUTOFF, event=day_event.event)
    assert "classify" not in _tasks(stub_llm)


# --- degraded runs are not cached -------------------------------------------------------------------

def _fail_once(reply):
    """A reply function that raises a transient-looking error the first time, then answers."""
    state = {"n": 0}

    def call(prompt):
        state["n"] += 1
        if state["n"] == 1:
            raise TimeoutError("model call timed out")
        return reply

    return call


def test_classify_skipped_for_time_not_stored(day_event, stub_llm, monkeypatch):
    clock = {"t": 1000.0}
    monkeypatch.setattr(engine, "_now", lambda: clock["t"])

    def slow_compose(prompt):
        clock["t"] += 21.0
        return _compose(GOOD_DAY, SECOND_DAY)

    _script(stub_llm, slow_compose)
    first = _answer(DAY_Q, DAY_CUTOFF, event=day_event.event, interactive=True)
    assert first["sentences"] and "classify" not in _tasks(stub_llm)
    assert _stored() == []
    _answer(DAY_Q, DAY_CUTOFF, event=day_event.event)  # batch: full run
    assert _tasks(stub_llm).count("classify") == 1
    assert len(_stored()) == 1


def test_terms_call_raises_not_stored(day_event, stub_llm):
    _script(stub_llm, lambda prompt: _compose(GOOD_DAY))
    stub_llm.replies["terms"] = _fail_once(json.dumps({"terms": ["rescued"]}))
    first = _answer(DAY_Q, DAY_CUTOFF, event=day_event.event)
    assert [s["text"] for s in first["sentences"]] == [GOOD_DAY["text"]]
    assert _stored() == []
    _answer(DAY_Q, DAY_CUTOFF, event=day_event.event)
    assert _tasks(stub_llm).count("terms") == 2 and _tasks(stub_llm).count("compose") == 2
    assert len(_stored()) == 1


def test_rerank_call_raises_not_stored(day_event, stub_llm):
    _script(stub_llm, lambda prompt: _compose(GOOD_DAY))
    stub_llm.replies["rerank"] = _fail_once(json.dumps({"ids": ["day-06"]}))
    first = _answer(DAY_Q, DAY_CUTOFF, event=day_event.event)
    assert first["sentences"]
    assert _stored() == []
    _answer(DAY_Q, DAY_CUTOFF, event=day_event.event)
    assert _tasks(stub_llm).count("rerank") == 2
    assert len(_stored()) == 1


def test_classify_call_raises_not_stored(day_event, stub_llm):
    _script(stub_llm, lambda prompt: _compose(GOOD_DAY, SECOND_DAY),
            classify=_fail_once(json.dumps({"assign": {}})))
    _answer(DAY_Q, DAY_CUTOFF, event=day_event.event)
    assert _stored() == []
    _answer(DAY_Q, DAY_CUTOFF, event=day_event.event)
    assert _tasks(stub_llm).count("classify") == 2
    assert len(_stored()) == 1


def test_malformed_rerank_reply_stored(day_event, stub_llm):
    _script(stub_llm, lambda prompt: _compose(GOOD_DAY))
    stub_llm.replies["rerank"] = lambda prompt: "this is not JSON"
    first = _answer(DAY_Q, DAY_CUTOFF, event=day_event.event)
    assert first["sentences"]
    assert len(_stored()) == 1
    _answer(DAY_Q, DAY_CUTOFF, event=day_event.event)
    assert _tasks(stub_llm).count("rerank") == 1


def test_malformed_terms_reply_stored(day_event, stub_llm):
    _script(stub_llm, lambda prompt: _compose(GOOD_DAY))
    stub_llm.replies["terms"] = lambda prompt: json.dumps({"terms": "not a list"})
    _answer(DAY_Q, DAY_CUTOFF, event=day_event.event)
    assert len(_stored()) == 1


def test_interactive_out_of_time_before_search_terms(day_event, stub_llm, monkeypatch):
    ticks = iter([1000.0])
    monkeypatch.setattr(engine, "_now", lambda: next(ticks, 1031.0))
    _script(stub_llm, [_compose(GOOD_DAY)])
    result = _answer(DAY_Q, DAY_CUTOFF, event=day_event.event, interactive=True)
    assert result["notice"] == "The model could not produce an answer. Try again."
    assert stub_llm.calls == [] and _stored() == []


def test_interactive_out_of_time_before_rerank(day_event, stub_llm, monkeypatch):
    clock = {"t": 1000.0}
    monkeypatch.setattr(engine, "_now", lambda: clock["t"])

    def slow_terms(prompt):
        clock["t"] += 31.0
        return json.dumps({"terms": []})

    _script(stub_llm, [_compose(GOOD_DAY)])
    stub_llm.replies["terms"] = slow_terms
    result = _answer(DAY_Q, DAY_CUTOFF, event=day_event.event, interactive=True)
    assert result["notice"] == "The model could not produce an answer. Try again."
    assert result["retrieved"]
    assert _tasks(stub_llm) == ["terms"] and _stored() == []


def test_limits_read_fresh_each_lookup(monkeypatch):
    clock = {"t": 1000.0}
    monkeypatch.setattr(engine, "_now", lambda: clock["t"])
    limits = engine._Limits(1000.0)
    assert dict(limits) == {"attempts": 2, "timeout_s": 20.0, "deadline_s": 30.0}
    list(limits.keys())
    clock["t"] += 15.0
    assert limits.get("deadline_s") == 15.0 and limits["timeout_s"] == 15.0
    assert {**limits}["deadline_s"] == 15.0
    with pytest.raises(KeyError):
        limits["max_tokens"]


# --- interactive limits ----------------------------------------------------------------------------

def _record_kwargs(monkeypatch, stub):
    seen = []

    def complete(prompt, **kwargs):
        seen.append(kwargs)
        return stub(prompt, **kwargs)

    monkeypatch.setattr(llm, "complete", complete)
    return seen


def test_interactive_limits_passed_to_model_calls(day_event, stub_llm, monkeypatch):
    seen = _record_kwargs(monkeypatch, stub_llm)
    _script(stub_llm, [_compose(GOOD_DAY, SECOND_DAY)])
    _valid(engine.answer(DAY_Q, DAY_CUTOFF, event=day_event.event, interactive=True), DAY_CUTOFF)
    assert len(seen) == 4
    for kwargs in seen:
        assert kwargs["attempts"] == 2
        assert 0 < kwargs["timeout_s"] <= 20
        assert 0 < kwargs["deadline_s"] <= 30
        assert kwargs["timeout_s"] <= kwargs["deadline_s"]


def test_batch_calls_keep_default_retries(day_event, stub_llm, monkeypatch):
    seen = _record_kwargs(monkeypatch, stub_llm)
    _script(stub_llm, [_compose(GOOD_DAY)])
    _answer(DAY_Q, DAY_CUTOFF, event=day_event.event)
    assert seen and all(not ({"attempts", "timeout_s", "deadline_s"} & set(k)) for k in seen)


def test_interactive_budget_shrinks_across_compose_retry(day_event, stub_llm, monkeypatch):
    seen = _record_kwargs(monkeypatch, stub_llm)
    clock = {"t": 1000.0}
    monkeypatch.setattr(engine, "_now", lambda: clock["t"])

    def slow_bad(prompt):
        clock["t"] += 12.0
        return "not json"

    _script(stub_llm, slow_bad)
    result = _answer(DAY_Q, DAY_CUTOFF, event=day_event.event, interactive=True)
    compose_calls = [k for k, (t, _) in zip(seen, stub_llm.calls) if t == "compose"]
    assert len(compose_calls) == 2
    assert compose_calls[1]["deadline_s"] <= 30 - 12
    assert result["notice"] == "The model could not produce an answer. Try again."


def test_interactive_skips_classify_after_20_seconds(day_event, stub_llm, monkeypatch):
    clock = {"t": 1000.0}
    monkeypatch.setattr(engine, "_now", lambda: clock["t"])

    def slow_compose(prompt):
        clock["t"] += 21.0
        return _compose(GOOD_DAY, SECOND_DAY)

    _script(stub_llm, slow_compose)
    result = _valid(engine.answer(DAY_Q, DAY_CUTOFF, event=day_event.event, interactive=True), DAY_CUTOFF)
    assert "classify" not in _tasks(stub_llm)
    assert result["sentences"]


def test_interactive_out_of_time_before_compose(day_event, stub_llm, monkeypatch):
    clock = {"t": 1000.0}
    monkeypatch.setattr(engine, "_now", lambda: clock["t"])

    def slow_rerank(prompt):
        clock["t"] += 31.0
        return json.dumps({"ids": ["day-06"]})

    _script(stub_llm, [_compose(GOOD_DAY)])
    stub_llm.replies["rerank"] = slow_rerank
    result = _valid(engine.answer(DAY_Q, DAY_CUTOFF, event=day_event.event, interactive=True), DAY_CUTOFF)
    assert result["notice"] == "The model could not produce an answer. Try again."
    assert "compose" not in _tasks(stub_llm)


# --- moment and selection --------------------------------------------------------------------------

def test_moment_wording_day_and_minute(day_event, minute_event, monkeypatch):
    from eval.testset import cutoff_phrase

    day = events.parse_cutoff("", DAY_CUTOFF)
    assert engine.moment(day_event.event, day) == cutoff_phrase(DAY_CUTOFF) == "the end of Monday, April 15, 1912"
    at = datetime(2023, 6, 21, 14, 0, tzinfo=timezone.utc)
    assert engine.moment(minute_event.event, at) == "10:00 a.m. EDT, Wednesday, June 21, 2023"
    late = datetime(2023, 6, 21, 2, 30, tzinfo=timezone.utc)
    assert engine.moment(minute_event.event, late) == "10:30 p.m. EDT, Tuesday, June 20, 2023"
    noon = datetime(2023, 6, 21, 16, 5, tzinfo=timezone.utc)
    assert engine.moment(minute_event.event, noon) == "12:05 p.m. EDT, Wednesday, June 21, 2023"
    spec = {k: v for k, v in events.EVENTS[minute_event.event].items()
            if k not in ("display_offset_hours", "display_tz_label")}
    monkeypatch.setitem(events.EVENTS, minute_event.event, spec)
    assert engine.moment(minute_event.event, at) == "2:00 p.m. UTC, Wednesday, June 21, 2023"


def test_titan_story_shows_eastern_time():
    spec = events.get("titan-2023")
    assert spec["display_offset_hours"] == -4 and spec["display_tz_label"] == "EDT"


def test_engine_selector_loads_pipeline(monkeypatch):
    from dateline import engine as selector

    monkeypatch.setattr(config, "ENGINE", "real")
    monkeypatch.setattr(config, "ENGINE_MODULE", "pipeline.engine")
    assert selector.get_answer() is engine.answer


# --- stored(): is an answer already in the cache? --------------------------------------------------

def test_stored_tells_whether_the_answer_is_cached(day_event, stub_llm):
    _script(stub_llm, [_compose(GOOD_DAY, SECOND_DAY)])
    assert engine.stored(DAY_Q, DAY_CUTOFF, event=day_event.event) is False
    engine.answer(DAY_Q, DAY_CUTOFF, event=day_event.event)
    calls = len(stub_llm.calls)
    assert engine.stored(DAY_Q, DAY_CUTOFF, event=day_event.event) is True
    assert engine.stored("were the titanic passengers safe", "1912-04-15", event=day_event.event) is True
    assert engine.stored(DAY_Q, date(1912, 4, 16), event=day_event.event) is False  # another moment
    assert len(stub_llm.calls) == calls  # never calls the model


def test_stored_is_false_without_an_index_or_answerable_question(day_event, stub_llm, monkeypatch):
    assert engine.stored("???", DAY_CUTOFF, event=day_event.event) is False
    assert engine.stored(DAY_Q, date(1912, 4, 13), event=day_event.event) is False  # before the first article
    monkeypatch.setattr(index, "index_path", lambda event: day_event.index.with_name("missing.sqlite"))
    assert engine.stored(DAY_Q, DAY_CUTOFF, event=day_event.event) is False


# --- ablation switches -----------------------------------------------------------------------------

def test_switches_default_on(monkeypatch):
    monkeypatch.delenv("DATELINE_TERM_EXPANSION", raising=False)
    monkeypatch.delenv("DATELINE_RERANK", raising=False)
    assert engine.switches() == {"term_expansion": True, "rerank": True}


@pytest.mark.parametrize("env", ["DATELINE_TERM_EXPANSION", "DATELINE_RERANK"])
def test_ablation_answers_never_share_the_normal_cache(day_event, stub_llm, monkeypatch, env):
    _script(stub_llm, [_compose(GOOD_DAY), _compose(GOOD_DAY), _compose(GOOD_DAY)])
    _answer(DAY_Q, DAY_CUTOFF, event=day_event.event)
    assert engine.stored(DAY_Q, DAY_CUTOFF, event=day_event.event)
    monkeypatch.setenv(env, "0")
    assert not engine.stored(DAY_Q, DAY_CUTOFF, event=day_event.event)
    _answer(DAY_Q, DAY_CUTOFF, event=day_event.event)
    assert _tasks(stub_llm).count("compose") == 2
    assert len(_stored()) == 2
    _answer(DAY_Q, DAY_CUTOFF, event=day_event.event)  # the ablation answer is cached under its own key
    monkeypatch.delenv(env)
    _answer(DAY_Q, DAY_CUTOFF, event=day_event.event)  # and the normal one is still there
    assert _tasks(stub_llm).count("compose") == 2


def test_term_expansion_off_skips_the_terms_call(day_event, stub_llm, monkeypatch):
    _script(stub_llm, [_compose(GOOD_DAY)])
    monkeypatch.setenv("DATELINE_TERM_EXPANSION", "0")
    _answer(DAY_Q, DAY_CUTOFF, event=day_event.event)
    assert "terms" not in _tasks(stub_llm)
    assert "compose" in _tasks(stub_llm)


def test_rerank_off_skips_the_rerank_call(day_event, stub_llm, monkeypatch):
    _script(stub_llm, [_compose(GOOD_DAY)])
    monkeypatch.setenv("DATELINE_RERANK", "0")
    result = _answer(DAY_Q, DAY_CUTOFF, event=day_event.event)
    assert "rerank" not in _tasks(stub_llm)
    assert result["sentences"]
