"""Per-story SQLite FTS5 index built from the standard corpus shape.

Rows are inserted in (available_at, article_id) order, so rowid order is availability order and a
cutoff becomes a single number: R = max_rowid(con, cutoff). Every search filters on `rowid <= R`.
"""
from __future__ import annotations

import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from dateline import config, events

REQUIRED = ["article_id", "paper", "date", "headline", "text", "url"]
OPTIONAL = ["available_at", "page", "state", "scan_url"]
BATCH = 5000
_WORD = re.compile(r"\w+")

SCHEMA = """
CREATE VIRTUAL TABLE fts USING fts5(headline, text, tokenize='porter unicode61');
CREATE TABLE meta(
    rowid INTEGER PRIMARY KEY, article_id TEXT UNIQUE, paper TEXT, date TEXT,
    available_at TEXT, page TEXT, url TEXT, scan_url TEXT, state TEXT, length INTEGER);
CREATE INDEX meta_available_at ON meta(available_at);
CREATE INDEX meta_rowid_length ON meta(rowid, length);
CREATE VIRTUAL TABLE fts_v USING fts5vocab(fts, instance);
"""


class CorpusError(ValueError):
    """The corpus file lacks required columns or has duplicate article IDs."""


class IndexMissing(FileNotFoundError):
    """The index for an event is missing or older than its corpus."""


def iso_z(dt: datetime) -> str:
    """UTC timestamp in the one format used for storing and comparing availability."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def index_path(event: str) -> Path:
    return Path(config.DATA_DIR) / f"index_{event}.sqlite"


def _corpus_path(event: str) -> Path:
    return Path(events.get(event)["corpus"])


def _text(value) -> str:
    return "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)


def _available(event: str, row: dict) -> str:
    """available_at when present, else the end of `date` (events.item_time semantics)."""
    stamp = _text(row.get("available_at")).strip()
    return iso_z(events.item_time(event, {"available_at": stamp, "date": _text(row["date"]).strip()[:10]}))


def build(event: str, corpus: Path | None = None, out: Path | None = None) -> Path:
    corpus = Path(corpus) if corpus is not None else _corpus_path(event)
    out = Path(out) if out is not None else index_path(event)
    df = pd.read_parquet(corpus)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise CorpusError(f"{corpus} is missing required columns: {', '.join(missing)}")
    if df["article_id"].duplicated().any():
        raise CorpusError(f"{corpus} has duplicate article_id values")

    rows = []
    for rec in df.to_dict("records"):
        rows.append((_available(event, rec), _text(rec["article_id"]), rec))
    rows.sort(key=lambda r: (r[0], r[1]))

    tmp = out.with_name(out.name + ".tmp")
    out.parent.mkdir(parents=True, exist_ok=True)
    if tmp.exists():
        tmp.unlink()
    con = sqlite3.connect(tmp)
    try:
        con.executescript(SCHEMA)
        fts_sql = "INSERT INTO fts(rowid, headline, text) VALUES (?, ?, ?)"
        meta_sql = "INSERT INTO meta VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        fts_batch, meta_batch = [], []
        for rowid, (avail, aid, rec) in enumerate(rows, start=1):
            headline, text = _text(rec["headline"]), _text(rec["text"])
            fts_batch.append((rowid, headline, text))
            meta_batch.append((
                rowid, aid, _text(rec["paper"]), _text(rec["date"])[:10], avail, _text(rec.get("page")),
                _text(rec["url"]), _text(rec.get("scan_url")), _text(rec.get("state")),
                len(_WORD.findall(f"{headline} {text}"))))
            if len(fts_batch) >= BATCH:
                con.executemany(fts_sql, fts_batch)
                con.executemany(meta_sql, meta_batch)
                fts_batch, meta_batch = [], []
        if fts_batch:
            con.executemany(fts_sql, fts_batch)
            con.executemany(meta_sql, meta_batch)
        con.execute("INSERT INTO fts(fts) VALUES('optimize')")
        con.commit()
    except BaseException:
        con.close()
        tmp.unlink(missing_ok=True)
        raise
    con.close()
    os.replace(tmp, out)
    return out


def connect(event: str) -> sqlite3.Connection:
    """Open the event's index read-only. Never builds: a missing or stale index raises IndexMissing."""
    path = index_path(event)
    command = f"python -m pipeline.index {event}"
    if not path.exists():
        raise IndexMissing(f"No index for {event!r} at {path}. Build it with: {command}")
    corpus = _corpus_path(event)
    if corpus.exists() and path.stat().st_mtime < corpus.stat().st_mtime:
        raise IndexMissing(f"The index for {event!r} is older than its corpus {corpus}. Rebuild it with: {command}")
    con = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def max_rowid(con: sqlite3.Connection, cutoff: datetime) -> int:
    """Largest rowid available at the cutoff (inclusive); 0 if nothing is."""
    row = con.execute("SELECT max(rowid) FROM meta WHERE available_at <= ?", (iso_z(cutoff),)).fetchone()
    return int(row[0] or 0)


_FIELDS = ["article_id", "paper", "date", "available_at", "page", "url", "scan_url", "state"]


def get_articles(con: sqlite3.Connection, ids: Iterable[str]) -> dict[str, dict]:
    ids = list(dict.fromkeys(ids))
    out: dict[str, dict] = {}
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        marks = ",".join("?" * len(chunk))
        sql = (f"SELECT m.{', m.'.join(_FIELDS)}, f.headline, f.text FROM meta m "
               f"JOIN fts f ON f.rowid = m.rowid WHERE m.article_id IN ({marks})")
        for row in con.execute(sql, chunk):
            out[row[0]] = {k: ("" if row[k] is None else row[k]) for k in row.keys()}
    return out


def main(argv: list[str] | None = None) -> None:
    names = list(sys.argv[1:] if argv is None else argv)
    if not names:
        names = [e for e, spec in events.EVENTS.items() if Path(spec["corpus"]).exists()]
        if not names:
            print("no corpus files found; nothing to index")
            return
    for event in names:
        started = time.time()
        path = build(event)
        con = sqlite3.connect(path)
        rows = con.execute("SELECT count(*) FROM meta").fetchone()[0]
        con.close()
        print(f"{event}: {rows} rows, {path.stat().st_size / 1e6:.1f} MB, {time.time() - started:.1f} s -> {path}")


if __name__ == "__main__":
    main()
