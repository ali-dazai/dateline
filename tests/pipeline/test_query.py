"""Search terms: question words at full weight, a few filtered additions at low weight."""
from __future__ import annotations

import json

import pytest

from dateline import llm
from pipeline import prompts, query
from pipeline.search import tokens

QUESTION = "Is everyone on the Titanic safe?"
MOMENT = "April 15, 1912"


def _reply(*terms):
    return lambda prompt: json.dumps({"terms": list(terms)})


def test_question_words_always_included(stub_llm):
    stub_llm.replies["terms"] = [json.dumps({"terms": ["rescued"]})]
    weights = query.search_terms(QUESTION, MOMENT)
    for word in tokens(QUESTION):
        assert weights[word] == query.QUESTION_WEIGHT
    assert weights["rescued"] == query.ADDED_WEIGHT


def test_added_names_and_numbers_dropped():
    added = ["rescued", "Carpathia", "1,500", "lost", "Halifax"]
    assert query.filter_added(QUESTION, added) == ["rescued", "lost"]


def test_capitalised_word_from_question_kept_and_multiword_split():
    assert query.filter_added(QUESTION, ["Safe", "Titanic"]) == []  # already in the question
    assert query.filter_added("Did the ship sink?", ["Sank", "Wreck sunk"]) == []  # capitalised, not in question
    assert query.filter_added("Did the ship sink?", ["went down", "ship's loss"]) == ["went", "loss"]


def test_at_most_eight_additions(stub_llm):
    stub_llm.replies["terms"] = [json.dumps({"terms": [f"word{chr(97 + i)}x" for i in range(12)]})]
    weights = query.search_terms(QUESTION, MOMENT)
    added = [w for w, x in weights.items() if x == query.ADDED_WEIGHT]
    assert len(added) == query.MAX_ADDED == 8


def test_model_failure_falls_back_to_question_words(stub_llm):
    stub_llm.replies["terms"] = [RuntimeError("down")]
    assert query.search_terms(QUESTION, MOMENT) == {w: 1.0 for w in tokens(QUESTION)}


def test_story_terms_added_at_low_weight(stub_llm):
    stub_llm.replies["terms"] = [json.dumps({"terms": []})]
    weights = query.search_terms(QUESTION, MOMENT, story_terms=["Titanic", "liner"])
    assert weights["liner"] == query.ADDED_WEIGHT
    assert weights["titanic"] == query.QUESTION_WEIGHT


def test_question_word_keeps_full_weight_when_also_added(stub_llm):
    stub_llm.replies["terms"] = [json.dumps({"terms": ["safe", "rescued"]})]
    weights = query.search_terms(QUESTION, MOMENT, story_terms=["everyone"])
    assert weights["safe"] == 1.0 and weights["everyone"] == 1.0 and weights["rescued"] == 0.3


def test_malformed_reply_ignored(stub_llm):
    base = {w: 1.0 for w in tokens(QUESTION)}
    stub_llm.replies["terms"] = [json.dumps({"terms": "rescued"}), json.dumps([1, 2]),
                                 json.dumps({"terms": [1, None]}), json.dumps({"other": []})]
    for _ in range(4):
        assert query.search_terms(QUESTION, MOMENT) == base


def test_empty_question_makes_no_call(stub_llm):
    assert query.search_terms("???", MOMENT) == {}
    assert query.search_terms("Why?", MOMENT, story_terms=["titanic"]) == {}
    assert stub_llm.calls == []


def test_question_terms_full_weight():
    assert query.question_terms(QUESTION) == {w: 1.0 for w in tokens(QUESTION)}


def test_llm_opts_forwarded(monkeypatch):
    seen = {}

    def fake(prompt, **kwargs):
        seen.update(kwargs)
        return {"terms": []}

    monkeypatch.setattr(llm, "complete_json", fake)
    query.search_terms(QUESTION, MOMENT, llm_opts={"attempts": 2, "deadline_s": 30.0})
    assert seen == {"max_tokens": query.TERMS_MAX_TOKENS, "attempts": 2, "deadline_s": 30.0}


def test_prompt_first_line_is_task_terms():
    assert prompts.TERMS.splitlines()[0] == "TASK: terms"
    text = prompts.TERMS.format(question=QUESTION, moment=MOMENT)
    assert QUESTION in text and MOMENT in text


def test_names_inside_multiword_terms_dropped():
    assert query.filter_added(QUESTION, ["lost at Halifax"]) == ["lost"]
    assert query.filter_added(QUESTION, ["survivors off Carpathia"]) == ["survivors"]
    assert query.filter_added(QUESTION, ["survivors near Carpathia"]) == ["survivors", "near"]
    assert query.filter_added(QUESTION, ["lost off (Carpathia)"]) == ["lost"]


def test_digit_inside_multiword_term_drops_only_that_word():
    assert query.filter_added(QUESTION, ["lost 1,500 souls"]) == ["lost", "souls"]


def test_on_fallback_reports_a_raised_call_only(stub_llm):
    reasons = []
    stub_llm.replies["terms"] = [ConnectionError("down"), "not json", json.dumps({"terms": 5})]
    assert query.search_terms(QUESTION, MOMENT, on_fallback=reasons.append) == query.question_terms(QUESTION)
    assert reasons == ["terms call failed: ConnectionError"]
    for _ in range(2):
        query.search_terms(QUESTION, MOMENT, on_fallback=reasons.append)
    assert reasons == ["terms call failed: ConnectionError"]


def test_expansion_off_makes_no_model_call(stub_llm, monkeypatch):
    stub_llm.replies["terms"] = [json.dumps({"terms": ["rescued"]})]
    monkeypatch.setenv(query.EXPANSION_ENV, "0")
    weights = query.search_terms(QUESTION, MOMENT, story_terms=["liner"])
    assert stub_llm.calls == []
    expected = query.question_terms(QUESTION)
    expected.setdefault("liner", query.ADDED_WEIGHT)
    assert weights == expected


def test_expansion_switch_reads_the_environment(monkeypatch):
    monkeypatch.delenv(query.EXPANSION_ENV, raising=False)
    assert query.expansion_on()
    for off in ("0", "false", "OFF", " no "):
        monkeypatch.setenv(query.EXPANSION_ENV, off)
        assert not query.expansion_on()
    monkeypatch.setenv(query.EXPANSION_ENV, "1")
    assert query.expansion_on()


@pytest.mark.parametrize("term", ["lost-Halifax", "lost \u201cHalifax\u201d", "lost/Halifax", "lost\u2014Halifax",
                                  "lost_Halifax"])
def test_name_glued_to_a_word_is_dropped(term):
    assert query.filter_added("Were the passengers saved?", [term]) == ["lost"]


def test_hyphenated_and_possessive_terms_unchanged():
    question = "Did the Titanic's lifeboats reach the ship?"
    assert query.filter_added(question, ["life-boats"]) == ["life", "boats"]
    assert query.filter_added(question, ["Titanic's lifeboats"]) == []
    assert query.filter_added(question, ["Titanic's rescue"]) == ["rescue"]
