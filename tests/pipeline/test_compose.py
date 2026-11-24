"""Compose: prompt contents, one retry on unparseable replies, and cleaning of the model's output."""
from __future__ import annotations

import json

import pytest

from dateline import llm
from pipeline import compose, prompts

QUESTION = "Is everyone on the Titanic safe?"
MOMENT = "April 15, 1912"
PASSAGES = {
    "A1": "The Titanic struck an iceberg, but wireless messages say that all passengers are safe.",
    "A2": "The steamer is believed to be unhurt and her people are in no danger.",
}
ARTICLES = {"A1": {"paper": "The New York Herald", "date": "1912-04-15"},
            "A2": {"paper": "The Daily Missoulian", "date": "1912-04-15"}}


def _sentence(**over):
    s = {"text": "The Herald reported all passengers safe.",
         "quote": "wireless messages say that all passengers are safe", "article_id": "A1", "claim": "all safe"}
    s.update(over)
    return s


def _reply(**over):
    r = {"sentences": [_sentence()], "claims": {"A1": "all safe", "A2": None}, "figures": []}
    r.update(over)
    return json.dumps(r)


def _run(stub_llm, reply, **kw):
    stub_llm.replies["compose"] = reply if isinstance(reply, list) else [reply]
    return compose.compose(QUESTION, MOMENT, PASSAGES, ARTICLES, **kw)


def test_valid_reply_parsed(stub_llm):
    out = _run(stub_llm, _reply())
    assert out["sentences"] == [_sentence()]
    assert out["claims"] == {"A1": "all safe", "A2": None}
    assert out["figures"] == []
    assert len(stub_llm.calls) == 1


def test_retry_once_with_parser_error(stub_llm):
    out = _run(stub_llm, ["not json at all", _reply()])
    assert len(out["sentences"]) == 1
    assert len(stub_llm.calls) == 2
    first, second = stub_llm.calls[0][1], stub_llm.calls[1][1]
    assert second.startswith(first)
    assert "Your previous reply could not be parsed:" in second
    assert second.endswith("Reply with JSON only.")


def test_two_bad_replies_raise(stub_llm):
    with pytest.raises(compose.ComposeError):
        _run(stub_llm, ["nope", json.dumps({"sentences": "x"})])
    assert len(stub_llm.calls) == 2


def test_non_dict_reply_is_retried(stub_llm, monkeypatch):
    replies = [[1, 2], json.loads(_reply())]
    monkeypatch.setattr(llm, "complete_json", lambda prompt, **kw: replies.pop(0))
    out = compose.compose(QUESTION, MOMENT, PASSAGES, ARTICLES)
    assert len(out["sentences"]) == 1 and not replies


def test_missing_fields_dropped(stub_llm):
    bad = [{"text": "No quote.", "article_id": "A1"}, {"quote": "q", "article_id": "A1"},
           {"text": "t", "quote": "q"}, {"text": 5, "quote": "q", "article_id": "A1"}, "str"]
    out = _run(stub_llm, _reply(sentences=[*bad, _sentence()]))
    assert out["sentences"] == [_sentence()]


def test_prompt_states_moment_and_rules(stub_llm):
    _run(stub_llm, _reply())
    prompt = stub_llm.calls[0][1]
    assert prompt.splitlines()[0] == "TASK: compose"
    assert MOMENT in prompt and QUESTION in prompt
    assert "Never state or hint at what happened later" in prompt
    assert "6 to 40 words" in prompt
    assert "No paper yet reports whether" in prompt
    assert "{" not in prompts.COMPOSE.format(question="q", moment="m", passages="p").split("Reply with JSON only")[0]


def test_cleaning_drops_bad_sentences_and_caps_at_six(stub_llm):
    many = [_sentence(text=f"  Sentence {i}.  ", quote=f" quote {i} ") for i in range(9)]
    out = _run(stub_llm, _reply(sentences=many))
    assert len(out["sentences"]) == 6
    assert out["sentences"][0]["text"] == "Sentence 0." and out["sentences"][0]["quote"] == "quote 0"
    no_claim = _sentence()
    del no_claim["claim"]
    out = _run(stub_llm, _reply(sentences=[no_claim]))
    assert out["sentences"][0]["claim"] is None


def test_claims_limited_to_given_ids(stub_llm):
    out = _run(stub_llm, _reply(claims={"A1": "x", "ZZ": "y", "A2": 7}))
    assert out["claims"] == {"A1": "x"}
    out = _run(stub_llm, _reply(claims=["bad"]))
    assert out["claims"] == {}


def test_figures_parsed_and_filtered(stub_llm):
    figs = [{"paper": "The New York Herald", "article_id": "A1", "value": "1,500", "unit": "lives", "claim": "c"},
            {"article_id": "A2", "value": 675},
            {"paper": "X", "article_id": "ZZ", "value": 3},
            {"paper": "X", "article_id": "A1", "value": "many"},
            {"paper": "X", "article_id": "A1", "value": True}, "junk"]
    out = _run(stub_llm, _reply(figures=figs))
    assert out["figures"] == [
        {"paper": "The New York Herald", "article_id": "A1", "value": 1500.0, "unit": "lives", "claim": "c"},
        {"paper": "The Daily Missoulian", "article_id": "A2", "value": 675.0, "unit": "", "claim": None},
    ]


def test_network_error_raises_compose_error_without_retry(stub_llm):
    with pytest.raises(compose.ComposeError):
        _run(stub_llm, [TimeoutError("slow"), _reply()])
    assert len(stub_llm.calls) == 1


def test_retry_uses_no_cache(monkeypatch):
    seen = []

    def fake(prompt, **kw):
        seen.append(kw)
        if len(seen) == 1:
            raise json.JSONDecodeError("bad", "", 0)
        return json.loads(_reply())

    monkeypatch.setattr(llm, "complete_json", fake)
    compose.compose(QUESTION, MOMENT, PASSAGES, ARTICLES)
    assert seen[0].get("max_tokens") == compose.COMPOSE_MAX_TOKENS == 1500
    assert "use_cache" not in seen[0]
    assert seen[1]["use_cache"] is False


def test_passages_rendered_with_delimiters(stub_llm):
    _run(stub_llm, _reply())
    prompt = stub_llm.calls[0][1]
    a = "<<<ARTICLE id=1 paper=The New York Herald date=1912-04-15>>>\n" + PASSAGES["A1"] + "\n<<<END>>>"
    b = "<<<ARTICLE id=2 paper=The Daily Missoulian date=1912-04-15>>>\n" + PASSAGES["A2"] + "\n<<<END>>>"
    assert a in prompt and b in prompt and prompt.index(a) < prompt.index(b)


def test_llm_opts_forwarded(monkeypatch):
    seen = {}

    def fake(prompt, **kw):
        seen.update(kw)
        return json.loads(_reply())

    monkeypatch.setattr(llm, "complete_json", fake)
    compose.compose(QUESTION, MOMENT, PASSAGES, ARTICLES, llm_opts={"deadline_s": 30.0, "attempts": 2})
    assert seen["deadline_s"] == 30.0 and seen["attempts"] == 2 and seen["max_tokens"] == 1500


def test_unhashable_ids_and_huge_numbers_do_not_raise(stub_llm):
    figs = [{"paper": "X", "article_id": ["A1"], "value": 1}, {"paper": "X", "article_id": {"a": 1}, "value": 1},
            {"paper": "X", "article_id": "A1", "value": 10**400},
            {"paper": "X", "article_id": "A1", "value": "1e999"},
            {"paper": "X", "article_id": "A1", "value": 7}]
    out = _run(stub_llm, _reply(figures=figs))
    assert [f["value"] for f in out["figures"]] == [7.0]


def test_blank_claims_become_none_and_claims_are_stripped(stub_llm):
    out = _run(stub_llm, _reply(sentences=[_sentence(claim="   ")], claims={"A1": "  safe  ", "A2": " "},
                                figures=[{"article_id": "A1", "value": 1, "claim": "  "}]))
    assert out["sentences"][0]["claim"] is None
    assert out["claims"] == {"A1": "safe", "A2": None}
    assert out["figures"][0]["claim"] is None


def test_short_labels_mapped_back_to_article_ids(stub_llm):
    reply = json.dumps({
        "sentences": [{"text": "The Herald reported that all passengers are safe.",
                       "quote": "wireless messages say that all passengers are safe", "article_id": "1",
                       "claim": "all safe"},
                      {"text": "The Missoulian reported no danger.", "quote": "her people are in no danger",
                       "article_id": 2, "claim": None},
                      {"text": "Unknown.", "quote": "some words that are quoted here", "article_id": "9",
                       "claim": None}],
        "claims": {"1": "all safe", "A2": "no danger"},
        "figures": [{"paper": "", "article_id": "2", "value": 3, "unit": "", "claim": None}]})
    out = _run(stub_llm, reply)
    assert [s["article_id"] for s in out["sentences"]] == ["A1", "A2", "9"]
    assert out["claims"] == {"A1": "all safe", "A2": "no danger"}
    assert [f["article_id"] for f in out["figures"]] == ["A2"]
