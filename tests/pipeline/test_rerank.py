"""Re-rank: kept IDs first, then the model's choices that exist, with a clean fallback."""
from __future__ import annotations

import json

from dateline import llm
from pipeline import prompts, rerank
from pipeline.search import Hit

QUESTION = "Is everyone on the Titanic safe?"
MOMENT = "April 15, 1912"


def _hits(n: int = 15) -> list[Hit]:
    return [Hit(f"A{i:02d}", 10.0 - i * 0.1, "The Evening Star", "1912-04-15", "1912-04-15T23:59:59Z",
                f"HEADLINE {i}") for i in range(n)]


def _openings(hits):
    return {h.article_id: f"opening of {h.article_id}" for h in hits}



def test_rerank_keeps_top_three_bm25(stub_llm):
    hits = _hits()
    stub_llm.replies["rerank"] = [json.dumps({"ids": ["A09", "A08"]})]
    out = rerank.rerank(QUESTION, MOMENT, hits, _openings(hits), keep_ids=["A00", "A01", "A02"])
    assert out == ["A00", "A01", "A02", "A09", "A08"]


def test_rerank_drops_unknown_ids(stub_llm):
    hits = _hits()
    stub_llm.replies["rerank"] = [json.dumps({"ids": ["X999", "A05", "A05", 70, None, True]})]
    out = rerank.rerank(QUESTION, MOMENT, hits, _openings(hits), keep_ids=["A00"])
    assert out == ["A00", "A05"]


def test_rerank_lists_short_labels_and_maps_them_back(stub_llm):
    hits = _hits()
    stub_llm.replies["rerank"] = [json.dumps({"ids": ["10", 9, "[3]", "A05"]})]
    out = rerank.rerank(QUESTION, MOMENT, hits, _openings(hits), keep_ids=["A00"])
    assert out == ["A00", "A09", "A08", "A02", "A05"]
    prompt = stub_llm.calls[0][1]
    assert "[1] The Evening Star" in prompt and "[A00]" not in prompt


def test_rerank_malformed_reply_falls_back(stub_llm):
    hits = _hits()
    keep = ["A03", "A01", "A02"]
    expected = keep + [h.article_id for h in hits if h.article_id not in keep][:7]
    for reply in ["sure! here:", json.dumps([1]), json.dumps({"ids": "A05"}), json.dumps({"x": 1})]:
        stub_llm.replies["rerank"] = [reply]
        assert rerank.rerank(QUESTION, MOMENT, hits, _openings(hits), keep_ids=keep) == expected
    stub_llm.replies["rerank"] = [RuntimeError("down")]
    assert rerank.rerank(QUESTION, MOMENT, hits, _openings(hits), keep_ids=keep) == expected


def test_rerank_empty_hits_no_call(stub_llm):
    assert rerank.rerank(QUESTION, MOMENT, [], {}, keep_ids=["A00"]) == []
    assert stub_llm.calls == []


def test_rerank_respects_limit(stub_llm):
    hits = _hits()
    stub_llm.replies["rerank"] = [json.dumps({"ids": [h.article_id for h in hits]})]
    out = rerank.rerank(QUESTION, MOMENT, hits, _openings(hits), keep_ids=["A00", "A01", "A02"])
    assert len(out) == rerank.LIMIT == 10
    stub_llm.replies["rerank"] = [json.dumps({"ids": [h.article_id for h in hits]})]
    assert len(rerank.rerank(QUESTION, MOMENT, hits, _openings(hits), keep_ids=["A00"], limit=4)) == 4


def test_rerank_keep_ids_not_in_hits_ignored(stub_llm):
    hits = _hits(5)
    stub_llm.replies["rerank"] = [json.dumps({"ids": ["A04"]})]
    out = rerank.rerank(QUESTION, MOMENT, hits, _openings(hits), keep_ids=["ZZZ", "A02", "A02"])
    assert out == ["A02", "A04"]


def test_rerank_prompt_lists_candidates_and_task_line(stub_llm):
    hits = _hits(3)
    stub_llm.replies["rerank"] = [json.dumps({"ids": []})]
    rerank.rerank(QUESTION, MOMENT, hits, _openings(hits), keep_ids=[])
    prompt = stub_llm.calls[0][1]
    assert prompt.splitlines()[0] == "TASK: rerank"
    assert QUESTION in prompt and MOMENT in prompt
    assert "[2] The Evening Star, 1912-04-15 — HEADLINE 1 — opening of A01" in prompt
    assert prompts.RERANK.splitlines()[0] == "TASK: rerank"


def test_rerank_llm_opts_forwarded(monkeypatch):
    seen = {}

    def fake(prompt, **kwargs):
        seen.update(kwargs)
        return {"ids": []}

    monkeypatch.setattr(llm, "complete_json", fake)
    hits = _hits(3)
    rerank.rerank(QUESTION, MOMENT, hits, _openings(hits), keep_ids=[], llm_opts={"attempts": 2, "deadline_s": 30.0})
    assert seen == {"max_tokens": rerank.RERANK_MAX_TOKENS, "attempts": 2, "deadline_s": 30.0}


def test_on_fallback_reports_a_raised_call_only(stub_llm):
    hits = _hits(5)
    reasons = []
    stub_llm.replies["rerank"] = [TimeoutError("slow"), "not json", json.dumps({"ids": "x"})]
    out = rerank.rerank(QUESTION, MOMENT, hits, _openings(hits), keep_ids=[], on_fallback=reasons.append)
    assert out == [h.article_id for h in hits]
    assert reasons == ["rerank call failed: TimeoutError"]
    for _ in range(2):  # malformed replies fall back too, but are real replies: not reported
        rerank.rerank(QUESTION, MOMENT, hits, _openings(hits), keep_ids=[], on_fallback=reasons.append)
    assert reasons == ["rerank call failed: TimeoutError"]


def test_rerank_without_model_keeps_search_order(stub_llm):
    hits = _hits(12)
    out = rerank.rerank(QUESTION, MOMENT, hits, _openings(hits), keep_ids=["A05"], use_model=False)
    assert stub_llm.calls == []
    assert out == ["A05", "A00", "A01", "A02", "A03", "A04", "A06", "A07", "A08", "A09"]


def test_rerank_switch_reads_the_environment(monkeypatch):
    monkeypatch.delenv(rerank.RERANK_ENV, raising=False)
    assert rerank.rerank_on()
    monkeypatch.setenv(rerank.RERANK_ENV, "0")
    assert not rerank.rerank_on()
