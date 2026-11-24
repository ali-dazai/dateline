import pytest

from dateline import llm
from pipeline import index
from tests.pipeline.conftest import build_subset


def test_build_subset_keeps_only_matching_rows(day_event):
    path = build_subset(day_event, lambda row: row["date"] <= "1912-04-15", "early")
    assert path != day_event.index
    import sqlite3
    count = sqlite3.connect(path).execute("SELECT count(*) FROM meta").fetchone()[0]
    assert count == int((day_event.df["date"] <= "1912-04-15").sum())
    assert index.connect(day_event.event)  # the full index is untouched


def test_stub_llm_routes_and_consumes_in_order(stub_llm):
    stub_llm.replies["terms"] = ["a", "b"]
    stub_llm.replies["rerank"] = lambda prompt: prompt.upper()
    stub_llm.replies["compose"] = [ValueError("boom")]
    assert llm.complete("TASK: terms\nx", json_mode=True, system="s", max_tokens=5) == "a"
    assert llm.complete("TASK: terms\ny") == "b"
    assert llm.complete("TASK: rerank\nz") == "TASK: RERANK\nZ"
    with pytest.raises(ValueError):
        llm.complete("TASK: compose\n")
    with pytest.raises(AssertionError, match="terms"):
        llm.complete("TASK: terms\n")
    with pytest.raises(AssertionError, match="classify"):
        llm.complete("TASK: classify\n")
    assert [t for t, _ in stub_llm.calls] == ["terms", "terms", "rerank", "compose", "terms", "classify"]
