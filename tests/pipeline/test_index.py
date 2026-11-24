import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from dateline import config
from pipeline import index
from tests.pipeline.conftest import write_corpus


def test_fts5_available():
    con = sqlite3.connect(":memory:")
    try:
        con.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
    except sqlite3.OperationalError as exc:
        pytest.fail(f"this Python's SQLite ({sqlite3.sqlite_version}) lacks FTS5: {exc}")


def test_iso_z_converts_to_utc_and_treats_naive_as_utc():
    assert index.iso_z(datetime(2023, 6, 21, 10, 0, 0)) == "2023-06-21T10:00:00Z"
    tz = timezone(timedelta(hours=-4))
    assert index.iso_z(datetime(2023, 6, 21, 6, 0, 0, tzinfo=tz)) == "2023-06-21T10:00:00Z"


def test_index_path_follows_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    assert index.index_path("x") == tmp_path / "index_x.sqlite"


def test_build_creates_index_with_all_articles(day_event):
    con = index.connect(day_event.event)
    assert con.execute("SELECT count(*) FROM meta").fetchone()[0] == len(day_event.df)
    hits = con.execute("SELECT count(*) FROM fts WHERE fts MATCH 'titanic'").fetchone()[0]
    expected = (day_event.df["headline"] + " " + day_event.df["text"]).str.contains("titanic", case=False).sum()
    assert hits == expected >= 10
    stemmed = con.execute("SELECT count(DISTINCT doc) FROM fts_v WHERE term = 'titan'").fetchone()[0]
    assert stemmed == expected  # fts_v holds the stemmed terms
    assert not day_event.index.with_name(day_event.index.name + ".tmp").exists()
    plan = " ".join(r[-1] for r in con.execute("EXPLAIN QUERY PLAN SELECT * FROM meta WHERE available_at <= 'x'"))
    assert "available_at" in plan


def test_porter_stemming_matches_inflections(day_event):
    con = index.connect(day_event.event)
    assert con.execute("SELECT count(*) FROM fts WHERE fts MATCH 'sinking'").fetchone()[0] >= 1
    assert con.execute("SELECT count(*) FROM fts WHERE fts MATCH 'sinks'").fetchone()[0] >= 1


def test_length_counts_word_tokens(day_event):
    con = index.connect(day_event.event)
    row = con.execute("SELECT length FROM meta WHERE article_id = 'day-15'").fetchone()
    src = day_event.df.set_index("article_id").loc["day-15"]
    import re
    assert row[0] == len(re.findall(r"\w+", src["headline"] + " " + src["text"]))


def test_available_at_defaults_to_end_of_date(day_event):
    con = index.connect(day_event.event)
    row = con.execute("SELECT available_at FROM meta WHERE article_id = 'day-02'").fetchone()
    assert row[0] == "1912-04-15T23:59:59Z"


def test_rowids_follow_availability_order(minute_event):
    con = index.connect(minute_event.event)
    rows = con.execute("SELECT available_at, article_id FROM meta ORDER BY rowid").fetchall()
    assert [tuple(r) for r in rows] == sorted(tuple(r) for r in rows)
    fts_ids = {r[0] for r in con.execute("SELECT rowid FROM fts")}
    assert fts_ids == {r[0] for r in con.execute("SELECT rowid FROM meta")}


def test_max_rowid_includes_item_exactly_at_cutoff(minute_event):
    con = index.connect(minute_event.event)
    at = datetime(2023, 6, 21, 10, 0, 0, tzinfo=timezone.utc)
    r = index.max_rowid(con, at)
    last = con.execute("SELECT article_id, available_at FROM meta WHERE rowid = ?", (r,)).fetchone()
    assert last["available_at"] == "2023-06-21T10:00:00Z"
    assert last["article_id"] == "min-06"
    r_before = index.max_rowid(con, at - timedelta(seconds=1))
    assert r_before == r - 1
    nxt = con.execute("SELECT article_id FROM meta WHERE rowid = ?", (r + 1,)).fetchone()
    assert nxt["article_id"] == "min-07"  # 10:01 is in the same hour but excluded at 10:00


def test_max_rowid_zero_before_first_article(minute_event):
    con = index.connect(minute_event.event)
    assert index.max_rowid(con, datetime(2023, 6, 1, tzinfo=timezone.utc)) == 0


def test_edited_later_item_counts_from_available_at(minute_event):
    con = index.connect(minute_event.event)
    rid = con.execute("SELECT rowid FROM meta WHERE article_id = 'min-11'").fetchone()[0]
    assert con.execute("SELECT date FROM meta WHERE rowid = ?", (rid,)).fetchone()[0] == "2023-06-20"
    assert index.max_rowid(con, datetime(2023, 6, 23, 8, 59, 59, tzinfo=timezone.utc)) < rid
    assert index.max_rowid(con, datetime(2023, 6, 23, 9, 0, 0, tzinfo=timezone.utc)) == rid


def test_missing_columns_raise_corpus_error(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    df = pd.DataFrame([{"article_id": "a", "paper": "p", "date": "1912-04-15", "headline": "h", "text": "t"}])
    path = write_corpus(df, tmp_path / "c.parquet")
    with pytest.raises(index.CorpusError, match="url"):
        index.build("titanic-1912", corpus=path, out=tmp_path / "o.sqlite")


def test_connect_missing_index_raises_with_command(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    with pytest.raises(index.IndexMissing, match=r"python -m pipeline\.index titanic-1912") as exc:
        index.connect("titanic-1912")
    assert isinstance(exc.value, FileNotFoundError)


def _make_stale(fx):
    """Index 1000 s old, corpus 500 s old (both in the past, so a rebuild is newer than both)."""
    now = time.time()
    os.utime(fx.index, (now - 1000, now - 1000))
    os.utime(fx.corpus, (now - 500, now - 500))
    return now - 1000


def test_stale_index_raises_index_missing(day_event):
    old = _make_stale(day_event)
    with pytest.raises(index.IndexMissing, match=r"python -m pipeline\.index test-day"):
        index.connect(day_event.event)
    assert day_event.index.stat().st_mtime == pytest.approx(old, abs=1)  # connect() did not rebuild


def test_build_replaces_stale_index(day_event):
    _make_stale(day_event)
    stale_tmp = day_event.index.with_name(day_event.index.name + ".tmp")
    stale_tmp.write_bytes(b"junk")
    index.build(day_event.event)
    assert not stale_tmp.exists()
    con = index.connect(day_event.event)
    assert con.execute("SELECT count(*) FROM meta").fetchone()[0] == len(day_event.df)


def test_connect_is_read_only(day_event):
    con = index.connect(day_event.event)
    with pytest.raises(sqlite3.OperationalError):
        con.execute("DELETE FROM meta")


def test_get_articles_returns_requested_ids(day_event):
    con = index.connect(day_event.event)
    got = index.get_articles(con, ["day-02", "day-09", "nope"])
    assert set(got) == {"day-02", "day-09"}
    art = got["day-02"]
    assert set(art) == {"article_id", "paper", "date", "available_at", "page", "url", "scan_url",
                        "state", "headline", "text"}
    assert art["paper"] == "The New York Herald" and art["page"] == "1"
    assert "iceberg" in art["text"] and art["state"] == "New York"
    assert index.get_articles(con, []) == {}


def test_optional_columns_default_to_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    df = pd.DataFrame([{"article_id": "a", "paper": "p", "date": "1912-04-15", "headline": "h",
                        "text": "some text", "url": "u"}])
    corpus = write_corpus(df, tmp_path / "c.parquet")
    out = index.build("titanic-1912", corpus=corpus, out=tmp_path / "o.sqlite")
    con = sqlite3.connect(out)
    con.row_factory = sqlite3.Row
    art = index.get_articles(con, ["a"])["a"]
    assert art["page"] == art["state"] == art["scan_url"] == ""
    assert art["available_at"] == "1912-04-15T23:59:59Z"


def test_mixed_offset_and_int_page_are_normalised(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    df = pd.DataFrame([
        {"article_id": "a", "paper": "p", "date": "2023-06-21", "headline": "h", "text": "x", "url": "u",
         "available_at": "2023-06-21T12:00:00+02:00", "page": 7},
        {"article_id": "b", "paper": "p", "date": "2023-06-21", "headline": "h", "text": "y", "url": "u",
         "available_at": "", "page": 8},
    ])
    out = index.build("titan-2023", corpus=write_corpus(df, tmp_path / "c.parquet"), out=tmp_path / "o.sqlite")
    con = sqlite3.connect(out)
    rows = dict(con.execute("SELECT article_id, available_at FROM meta"))
    assert rows == {"a": "2023-06-21T10:00:00Z", "b": "2023-06-21T23:59:59Z"}
    assert con.execute("SELECT page FROM meta WHERE article_id='a'").fetchone()[0] == "7"
