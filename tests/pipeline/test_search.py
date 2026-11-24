import math
import sqlite3
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from pipeline import index, search
from tests.pipeline.conftest import build_subset, write_corpus

UTC = timezone.utc
END_OF_15 = datetime(1912, 4, 15, 23, 59, 59, tzinfo=UTC)

TERM_LISTS = [
    {"titanic": 1.0},
    {"iceberg": 1.0, "passengers": 1.0, "safe": 1.0},
    {"ship": 1.0, "passengers": 1.0, "saved": 1.0, "lost": 1.0},
    {"carpathia": 1.0, "survivors": 0.3, "liner": 0.3},
    {"white": 1.0, "star": 1.0, "line": 1.0, "wheat": 0.3, "cattle": 0.3},
]


def _open(path) -> sqlite3.Connection:
    return sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)


def _same(a: list[search.Hit], b: list[search.Hit]) -> None:
    assert [h.article_id for h in a] == [h.article_id for h in b]
    for x, y in zip(a, b):
        assert x.score == pytest.approx(y.score, abs=1e-9)
        assert (x.paper, x.date, x.available_at, x.headline) == (y.paper, y.date, y.available_at, y.headline)


def test_tokens_strip_fts_syntax():
    assert search.tokens('Titanic" OR NEAR(* -iceberg ^') == ["titanic", "near", "iceberg"]  # 'or' is a stop word


def test_tokens_lowercase_drop_short_and_duplicates():
    assert search.tokens("Ship ship, a 1912 x SHIP the Liner") == ["ship", "1912", "liner"]
    assert search.tokens("Why? ??? \U0001F6A2") == []
    assert "or" in search.STOP_WORDS and "and" in search.STOP_WORDS and "near" not in search.STOP_WORDS


def test_stem_matches_index(day_event):
    con = index.connect(day_event.event)
    assert search.stem("Titanic") == "titan"
    term = search.stem("sinking")
    via_vocab = {r[0] for r in con.execute("SELECT DISTINCT doc FROM fts_v WHERE term = ?", (term,))}
    via_match = {r[0] for r in con.execute("SELECT rowid FROM fts WHERE fts MATCH 'sinking'")}
    assert via_vocab == via_match and via_vocab
    assert search.stem("???") == ""


def test_stats_cover_rows_up_to_r(day_event):
    con = index.connect(day_event.event)
    r = index.max_rowid(con, datetime(1912, 4, 14, 23, 59, 59, tzinfo=UTC))
    assert r == 1
    n, avgdl = search.stats(con, r)
    assert n == 1
    assert avgdl == con.execute("SELECT length FROM meta WHERE rowid = 1").fetchone()[0]
    n, avgdl = search.stats(con, index.max_rowid(con, END_OF_15))
    assert n == 8


def test_bm25_matches_reference(day_event, tmp_path):
    df = pd.DataFrame([
        {"article_id": "a", "paper": "P", "date": "1912-04-15", "headline": "ice", "text": "ice ice berg", "url": "u"},
        {"article_id": "b", "paper": "P", "date": "1912-04-15", "headline": "ship", "text": "ship ice", "url": "u"},
        {"article_id": "c", "paper": "P", "date": "1912-04-15", "headline": "harbor",
         "text": "calm harbor water today", "url": "u"},
    ])
    path = index.build(day_event.event, corpus=write_corpus(df, tmp_path / "ref.parquet"), out=tmp_path / "ref.sqlite")
    hits = search.search(_open(path), {"ice": 1.0, "ship": 1.0}, END_OF_15)

    k1, b, n, avgdl = 1.2, 0.75, 3, 4.0  # lengths 4, 3, 5

    def part(tf, length):
        return tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / avgdl))

    idf_ice = math.log((n - 2 + 0.5) / (2 + 0.5) + 1)
    idf_ship = math.log((n - 1 + 0.5) / (1 + 0.5) + 1)
    expected = {"a": idf_ice * part(3, 4), "b": idf_ice * part(1, 3) + idf_ship * part(2, 3)}
    assert [h.article_id for h in hits] == sorted(expected, key=lambda i: -expected[i])
    for h in hits:
        assert h.score == pytest.approx(expected[h.article_id], abs=1e-12)


def test_hits_carry_meta_fields(day_event):
    con = index.connect(day_event.event)
    hits = search.search(con, {"wheat": 1.0}, END_OF_15)
    assert hits == [search.Hit(article_id="day-07", score=hits[0].score, paper="The Daily Missoulian",
                               date="1912-04-15", available_at="1912-04-15T23:59:59Z", headline="Local Markets")]


def test_never_returns_articles_after_cutoff(day_event):
    con = index.connect(day_event.event)
    start = datetime(1912, 4, 13, 12, 0, 0, tzinfo=UTC)
    returned = 0
    for i in range(20):
        cutoff = start + timedelta(hours=4 * i)
        for terms in TERM_LISTS:
            hits = search.search(con, terms, cutoff)
            returned += len(hits)
            assert all(h.available_at <= index.iso_z(cutoff) for h in hits)
    assert returned > 0


def test_scores_identical_without_later_articles(day_event):
    full = index.connect(day_event.event)
    sub = _open(build_subset(day_event, lambda row: row["date"] <= "1912-04-15", "upto15"))
    for terms in TERM_LISTS:
        a = search.search(full, terms, END_OF_15)
        b = search.search(sub, terms, END_OF_15)
        assert a
        _same(a, b)


def test_pool_fallback_is_date_local(day_event):
    full = index.connect(day_event.event)
    sub = _open(build_subset(day_event, lambda row: row["date"] <= "1912-04-15", "pool15"))
    for terms in TERM_LISTS:
        for pool in (1, 2, 5000):
            _same(search.search(full, terms, END_OF_15, pool=pool), search.search(sub, terms, END_OF_15, pool=pool))
        _same(search.search(full, terms, END_OF_15, pool=5000), search.search(full, terms, END_OF_15))
    small = search.search(full, TERM_LISTS[2], END_OF_15, pool=1)
    assert len(small) == 1


def test_minute_cutoff_respects_available_at(minute_event):
    con = index.connect(minute_event.event)
    terms = {"titan": 1.0, "submersible": 1.0, "search": 1.0, "wreckage": 1.0}
    cutoff = datetime(2023, 6, 21, 9, 0, 0, tzinfo=UTC)
    hits = search.search(con, terms, cutoff)
    assert hits and all(h.available_at <= "2023-06-21T09:00:00Z" for h in hits)
    late = search.search(con, {"wreckage": 1.0}, datetime(2023, 6, 22, 23, 0, 0, tzinfo=UTC))
    assert "min-11" not in [h.article_id for h in late]  # dated 06-20 but available 06-23
    later = search.search(con, {"wreckage": 1.0}, datetime(2023, 6, 23, 9, 0, 0, tzinfo=UTC))
    assert "min-11" in [h.article_id for h in later]


def test_article_exactly_at_cutoff_returned(minute_event):
    con = index.connect(minute_event.event)
    hits = search.search(con, {"oxygen": 1.0}, datetime(2023, 6, 21, 10, 0, 0, tzinfo=UTC))
    assert [h.article_id for h in hits] == ["min-06"]


def test_article_one_minute_after_cutoff_in_same_hour_excluded(minute_event):
    con = index.connect(minute_event.event)
    terms = {"french": 1.0, "robot": 1.0}
    assert search.search(con, terms, datetime(2023, 6, 21, 10, 0, 0, tzinfo=UTC)) == []
    assert search.search(con, terms, datetime(2023, 6, 21, 10, 0, 59, tzinfo=UTC)) == []
    hits = search.search(con, terms, datetime(2023, 6, 21, 10, 1, 0, tzinfo=UTC))
    assert [h.article_id for h in hits] == ["min-07"]


def test_empty_terms_return_empty(day_event):
    con = index.connect(day_event.event)
    assert search.search(con, {}, END_OF_15) == []
    assert search.search(con, {'"*(': 1.0}, END_OF_15) == []
    assert search.search(con, {"zzyzx": 1.0}, END_OF_15) == []
    assert search.search(con, {"titanic": 1.0}, datetime(1912, 4, 1, tzinfo=UTC)) == []  # before the first article


def test_weights_scale_scores(day_event):
    con = index.connect(day_event.event)
    one = {h.article_id: h.score for h in search.search(con, {"titanic": 1.0}, END_OF_15)}
    low = {h.article_id: h.score for h in search.search(con, {"titanic": 0.3}, END_OF_15)}
    assert one.keys() == low.keys() and one
    for aid in one:
        assert low[aid] == pytest.approx(0.3 * one[aid], abs=1e-12)
    ice = {h.article_id: h.score for h in search.search(con, {"iceberg": 1.0}, END_OF_15)}
    both = {h.article_id: h.score for h in search.search(con, {"titanic": 1.0, "iceberg": 0.3}, END_OF_15)}
    for aid in both:
        assert both[aid] == pytest.approx(one.get(aid, 0.0) + 0.3 * ice.get(aid, 0.0), abs=1e-12)


def test_duplicate_stems_merge_by_max_weight(day_event):
    con = index.connect(day_event.event)
    merged = search.search(con, {"sinks": 0.3, "sinking": 1.0}, datetime(1912, 4, 16, 23, 59, 59, tzinfo=UTC))
    single = search.search(con, {"sinking": 1.0}, datetime(1912, 4, 16, 23, 59, 59, tzinfo=UTC))
    _same(merged, single)


def test_more_than_12_terms_capped(day_event):
    con = index.connect(day_event.event)
    filler = {f"zq{chr(97 + i)}": 1.0 for i in range(12)}  # twelve words that match nothing
    assert search.search(con, {**filler, "baseball": 0.3}, END_OF_15 + timedelta(days=1)) == []
    assert search.search(con, {**filler, "baseball": 1.0}, END_OF_15 + timedelta(days=1)) == []  # tie: input order
    hits = search.search(con, {"baseball": 1.0, **filler}, END_OF_15 + timedelta(days=1))
    assert [h.article_id for h in hits] == ["day-15"]
    dupes = {"baseball": 0.3, **{f"zq{chr(97 + i)}": 1.0 for i in range(11)}, "baseballs": 0.2}
    assert [h.article_id for h in search.search(con, dupes, END_OF_15 + timedelta(days=1))] == ["day-15"]


def test_ties_break_by_article_id(day_event, tmp_path):
    df = pd.DataFrame([
        {"article_id": aid, "paper": "P", "date": day, "headline": "ice", "text": "ship harbor", "url": "u"}
        for aid, day in [("z", "1912-04-14"), ("m", "1912-04-15"), ("a", "1912-04-15")]
    ])
    path = index.build(day_event.event, corpus=write_corpus(df, tmp_path / "tie.parquet"), out=tmp_path / "tie.sqlite")
    con = _open(path)
    assert [h.article_id for h in search.search(con, {"ice": 1.0}, END_OF_15)] == ["a", "m", "z"]
    assert [h.article_id for h in search.search(con, {"ice": 1.0}, END_OF_15, k=2)] == ["a", "m"]


def test_length_scans_use_covering_index(day_event):
    # meta rows are wide (URLs); N, average length and the length table must not read them
    con = index.connect(day_event.event)
    for sql in ["SELECT count(*), avg(length) FROM meta WHERE rowid <= ?",
                "SELECT length FROM meta WHERE rowid <= ? ORDER BY rowid"]:
        plan = " ".join(r[-1] for r in con.execute("EXPLAIN QUERY PLAN " + sql, (5,)))
        assert "COVERING INDEX" in plan, plan


def test_stem_recovers_after_an_error(monkeypatch):
    con = sqlite3.connect(":memory:", check_same_thread=False)
    con.executescript("CREATE VIRTUAL TABLE s USING fts5(x, tokenize='porter unicode61');"
                      "CREATE VIRTUAL TABLE s_v USING fts5vocab(s, instance);")
    fail = {"on": True}

    class Flaky:
        def execute(self, sql, *args):
            if fail["on"] and sql.startswith("SELECT"):
                raise sqlite3.OperationalError("simulated")
            return con.execute(sql, *args)

    monkeypatch.setattr(search, "_STEM_CON", Flaky())
    with pytest.raises(sqlite3.OperationalError):
        search.stem.__wrapped__("passengers")
    fail["on"] = False
    assert search.stem.__wrapped__("passengers") == "passeng"
