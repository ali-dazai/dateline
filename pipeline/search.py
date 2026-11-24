"""Date-local BM25 over a story index.

FTS5's own bm25() takes its statistics (document count, average length, document frequency) from
the whole table, so articles published after the cutoff would change the ranking of earlier ones.
Here every statistic is computed over rows up to R = max_rowid(con, cutoff): rows are stored in
availability order, so nothing after the cutoff can influence a score.
"""
from __future__ import annotations

import heapq
import json
import math
import re
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache

from pipeline.index import max_rowid

K1 = 1.2
B = 0.75
MAX_TERMS = 12
POOL_TERMS = 3

_RUN = re.compile(r"[^\W_]+")

STOP_WORDS: frozenset[str] = frozenset("""
a about above after again against all am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have
having he her here hers herself him himself his how i if in into is it its itself just me more most
my myself no nor not of off on once only or other our ours ourselves out over own same she should so
some such than that the their theirs them themselves then there these they this those through to too
under until up very was we were what when where which while who whom why will with would you your
yours yourself yourselves
""".split())


@dataclass(frozen=True)
class Hit:
    article_id: str
    score: float
    paper: str
    date: str
    available_at: str
    headline: str


def tokens(text: str) -> list[str]:
    """Lowercase letter/digit runs of length >= 2, stop words removed, first occurrence order kept."""
    out: list[str] = []
    seen: set[str] = set()
    for word in _RUN.findall(text.lower()):
        if len(word) < 2 or word in STOP_WORDS or word in seen:
            continue
        seen.add(word)
        out.append(word)
    return out


_STEM_LOCK = threading.Lock()
_STEM_CON = sqlite3.connect(":memory:", check_same_thread=False)
_STEM_CON.executescript(
    "CREATE VIRTUAL TABLE s USING fts5(x, tokenize='porter unicode61');"
    "CREATE VIRTUAL TABLE s_v USING fts5vocab(s, instance);")


@lru_cache(maxsize=65536)
def stem(word: str) -> str:
    """The term the index stores for `word` (same tokenizer as the index); '' if it has no token."""
    with _STEM_LOCK:
        try:
            _STEM_CON.execute("INSERT INTO s(rowid, x) VALUES (1, ?)", (word,))
            row = _STEM_CON.execute("SELECT term FROM s_v ORDER BY offset LIMIT 1").fetchone()
        finally:  # an error must not leave row 1 behind, or every later INSERT would fail
            _STEM_CON.execute("DELETE FROM s")
    return row[0] if row else ""


def stats(con: sqlite3.Connection, R: int) -> tuple[int, float]:
    """Number of articles and their average length over rows up to R."""
    n, avg = con.execute("SELECT count(*), avg(length) FROM meta WHERE rowid <= ?", (R,)).fetchone()
    return int(n), float(avg or 0.0)


def _query_terms(terms: dict[str, float]) -> list[tuple[str, float, str]]:
    """(stem, weight, a raw word with that stem): duplicates merged by max weight, at most 12 kept."""
    merged: dict[str, list] = {}
    for raw, weight in terms.items():
        weight = float(weight)
        if weight <= 0:
            continue
        for run in _RUN.findall(str(raw).lower()):
            s = stem(run)
            if not s:
                continue
            if s not in merged:
                merged[s] = [weight, len(merged), run]
            elif weight > merged[s][0]:
                merged[s][0] = weight
    ranked = sorted(merged.items(), key=lambda kv: (-kv[1][0], kv[1][1]))[:MAX_TERMS]
    return [(s, w, raw) for s, (w, _order, raw) in ranked]


def _plain(con: sqlite3.Connection) -> sqlite3.Cursor:
    """A cursor returning plain tuples: index.connect() sets sqlite3.Row, which is slower per row."""
    cur = con.cursor()
    cur.row_factory = None
    return cur


def _ids(docs) -> str:
    return json.dumps(sorted(docs))


def _tf(con: sqlite3.Cursor, term: str, R: int, pool: str | None) -> list[tuple[int, int]]:
    if pool is None:
        sql = "SELECT doc, count(*) FROM fts_v WHERE term = ? AND doc <= ? GROUP BY doc"
        return con.execute(sql, (term, R)).fetchall()
    sql = ("SELECT doc, count(*) FROM fts_v WHERE term = ? AND doc <= ? "
           "AND doc IN (SELECT value FROM json_each(?)) GROUP BY doc")
    return con.execute(sql, (term, R, pool)).fetchall()


def _match_df(con: sqlite3.Cursor, raw: str, R: int) -> int:
    return con.execute("SELECT count(*) FROM fts WHERE fts MATCH ? AND rowid <= ?", (f'"{raw}"', R)).fetchone()[0]


def search(con: sqlite3.Connection, terms: dict[str, float], cutoff: datetime, k: int = 50,
           pool: int | None = None) -> list[Hit]:
    """Top k articles available at `cutoff` by BM25 with date-local statistics; ties by article_id.

    `terms` maps words to weights. With `pool` set, only up to `pool` of the most recent available
    articles containing one of the three date-locally rarest terms are scored (same formula)."""
    qterms = _query_terms(terms)
    if not qterms or k <= 0:
        return []
    con = _plain(con)
    R = max_rowid(con, cutoff)
    if R == 0:
        return []
    n, avgdl = stats(con, R)

    pool_ids: str | None = None
    dfs: dict[str, int] = {}
    if pool is not None:
        dfs = {s: _match_df(con, raw, R) for s, _w, raw in qterms}
        present = [(dfs[s], i, raw) for i, (s, _w, raw) in enumerate(qterms) if dfs[s] > 0]
        if not present:
            return []
        rarest = [raw for _df, _i, raw in sorted(present)[:POOL_TERMS]]
        expr = " OR ".join(f'"{raw}"' for raw in rarest)
        rows = con.execute("SELECT rowid FROM fts WHERE fts MATCH ? AND rowid <= ? ORDER BY rowid DESC LIMIT ?",
                           (expr, R, int(pool))).fetchall()
        pool_ids = _ids(r[0] for r in rows)

    postings: list[tuple[float, list[tuple[int, int]]]] = []
    for s, w, _raw in qterms:
        rows = _tf(con, s, R, pool_ids)
        df = dfs[s] if pool is not None else len(rows)
        if not rows or df == 0:
            continue
        idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
        postings.append((w * idf * (K1 + 1), rows))
    if not postings:
        return []

    docs = {doc for _c, rows in postings for doc, _tf_ in rows}
    if len(docs) * 5 > R and n == R:
        # Rowids are 1..R (build numbers rows consecutively), so a plain ordered scan is a lookup table
        # and is cheaper than fetching tens of thousands of rows by id.
        table = [r[0] for r in con.execute("SELECT length FROM meta WHERE rowid <= ? ORDER BY rowid", (R,))]
        norm = {doc: K1 * (1 - B + B * table[doc - 1] / avgdl) for doc in docs}
    else:
        lengths = con.execute("SELECT rowid, length FROM meta WHERE rowid IN (SELECT value FROM json_each(?))",
                              (_ids(docs),)).fetchall()
        norm = {doc: K1 * (1 - B + B * length / avgdl) for doc, length in lengths}

    scores: dict[int, float] = {}
    for coef, rows in postings:
        for doc, tf in rows:
            scores[doc] = scores.get(doc, 0.0) + coef * tf / (tf + norm[doc])

    top = heapq.nlargest(k, scores.values())
    floor = top[-1]
    candidates = {doc: sc for doc, sc in scores.items() if sc >= floor}
    aid = dict(con.execute("SELECT rowid, article_id FROM meta WHERE rowid IN (SELECT value FROM json_each(?))",
                           (_ids(candidates),)).fetchall())
    chosen = sorted(candidates, key=lambda doc: (-candidates[doc], aid[doc]))[:k]
    meta_sql = ("SELECT m.rowid, m.article_id, m.paper, m.date, m.available_at, f.headline FROM meta m "
                "JOIN fts f ON f.rowid = m.rowid WHERE m.rowid IN (SELECT value FROM json_each(?))")
    meta = {r[0]: r for r in con.execute(meta_sql, (_ids(chosen),))}
    return [Hit(article_id=meta[d][1], score=candidates[d], paper=meta[d][2] or "", date=meta[d][3] or "",
                available_at=meta[d][4] or "", headline=meta[d][5] or "") for d in chosen]
