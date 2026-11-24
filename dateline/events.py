"""The demo stories (events) and how cutoffs work for each (interface v2, docs/engine-interface.md).

Day stories (1912, 2008): the cutoff is a date and includes that whole day.
Minute story (2023): the cutoff is a UTC datetime; an item counts once its `available_at`
(fetch_guardian.py) is at or before the cutoff.
"""
from __future__ import annotations

import json
from datetime import date, datetime, time, timezone
from functools import lru_cache
from pathlib import Path

import pandas as pd

from . import config

DEFAULT_EVENT = "titanic-1912"

EVENTS: dict[str, dict] = {
    "titanic-1912": {
        "title": "Titanic",
        "subject": "the Titanic",  # as the plain comparison names the story ("About the Titanic ...")
        "dates": "April 10–20, 1912",
        "start": "1912-04-10",
        "end": "1912-04-20",
        "granularity": "day",
        "look": "sunday-plates",
        "corpus": config.CORPUS_PATH,
        "story_terms": ["titanic"],
        "source": "American Stories (Chronicling America)",
    },
    "lehman-2008": {
        "title": "Lehman Brothers",
        "subject": "Lehman Brothers",  # as the plain comparison names the story ("About the Titanic ...")
        "dates": "September 10–19, 2008",
        "start": "2008-09-10",
        "end": "2008-09-19",
        "granularity": "day",
        "look": "highlighter",
        "corpus": str(config.DATA_DIR / "articles_lehman-2008.parquet"),
        "story_terms": ["lehman"],
        "source": "The Guardian / The Observer (Guardian Open Platform)",
    },
    "titan-2023": {
        "title": "Titan submersible",
        "subject": "the Titan submersible",  # as the plain comparison names the story ("About the Titanic ...")
        "dates": "June 18–23, 2023",
        "start": "2023-06-18T00:00:00Z",
        "end": "2023-06-23T23:59:59Z",
        "granularity": "minute",
        "look": "abyss",
        "corpus": str(config.DATA_DIR / "articles_titan-2023.parquet"),
        "story_terms": ["titan", "submersible", "oceangate"],
        # The page shows US Eastern time; a fixed offset, as Windows Python has no time-zone database.
        "display_offset_hours": -4,
        "display_tz_label": "EDT",
        "source": "The Guardian / The Observer (Guardian Open Platform)",
    },
}


def get(event: str) -> dict:
    if event not in EVENTS:
        raise KeyError(f"unknown event {event!r}; choose from {sorted(EVENTS)}")
    return EVENTS[event]


def _utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def parse_cutoff(event: str, value: str | date | datetime) -> datetime:
    """Any cutoff -> a UTC datetime. A bare date means the END of that day (inclusive)."""
    if isinstance(value, datetime):
        return _utc(value)
    if isinstance(value, date):
        return datetime.combine(value, time(23, 59, 59), tzinfo=timezone.utc)
    text = str(value).strip()
    if len(text) == 10:
        return parse_cutoff(event, date.fromisoformat(text))
    return _utc(datetime.fromisoformat(text.replace("Z", "+00:00")))


def bounds(event: str) -> tuple[datetime, datetime]:
    spec = get(event)
    start = parse_cutoff(event, spec["start"])
    if len(spec["start"]) == 10:
        start = start.replace(hour=0, minute=0, second=0)
    return start, parse_cutoff(event, spec["end"])


def in_range(event: str, cutoff: datetime) -> bool:
    lo, hi = bounds(event)
    return lo <= cutoff <= hi


def item_time(event: str, row: dict | pd.Series) -> datetime:
    """When an article counts as available: available_at for Guardian items, end of day otherwise."""
    stamp = row.get("available_at") or row.get("published") or row.get("date")
    return parse_cutoff(event, stamp)


def cutoff_label(event: str, cutoff: datetime) -> str:
    if get(event)["granularity"] == "day":
        return cutoff.date().isoformat()
    return cutoff.strftime("%Y-%m-%dT%H:%MZ")


@lru_cache(maxsize=None)
def counts(event: str) -> list[dict]:
    """Articles per day (day stories) or per hour (minute story), for the day picker.

    Each bucket: {"bucket", "total", "story"}; story = items mentioning the story terms.
    Cached to data/counts_<event>.json because the 1912 table has 282k rows.
    """
    spec = get(event)
    cache = config.DATA_DIR / f"counts_{event}.json"
    corpus = Path(spec["corpus"])
    # Use the cached counts when they are current, or when the corpus is not on this machine
    # (the cache is tracked in git, so a fresh clone can still serve the day picker).
    if cache.exists() and (not corpus.exists() or cache.stat().st_mtime >= corpus.stat().st_mtime):
        return json.loads(cache.read_text())
    if not corpus.exists():
        return []
    cols = ["date", "headline", "text"] + (["available_at"] if spec["granularity"] == "minute" else [])
    df = pd.read_parquet(corpus, columns=cols)
    pattern = "|".join(spec["story_terms"])
    story = (df["headline"].fillna("") + " " + df["text"].fillna("")).str.contains(pattern, case=False, regex=True)
    if spec["granularity"] == "day":
        key = df["date"].astype(str).str[:10]
    else:
        key = pd.to_datetime(df["available_at"], utc=True).dt.strftime("%Y-%m-%dT%H:00Z")
    out = (pd.DataFrame({"bucket": key, "story": story})
           .groupby("bucket").agg(total=("story", "size"), story=("story", "sum")).reset_index())
    rows = [{"bucket": r.bucket, "total": int(r.total), "story": int(r.story)} for r in out.itertuples()]
    lo, hi = bounds(event)
    rows = [r for r in rows if lo <= parse_cutoff(event, r["bucket"]) and
            parse_cutoff(event, r["bucket"]).replace(hour=0, minute=0, second=0) <= hi]
    cache.write_text(json.dumps(rows))
    return rows
