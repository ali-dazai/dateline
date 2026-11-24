"""Fetch Guardian / Observer articles for the 2008 and 2023 demo stories.

Usage:  .venv\\Scripts\\python.exe scripts\\fetch_guardian.py [lehman-2008|titan-2023|all]
Needs GUARDIAN_API_KEY in .env (free developer key, ~5,000 calls/day, 12/second).

Writes data/articles_<event>.parquet with the same columns as the 1912 table plus:
  published  first-publication time, ISO 8601 UTC (e.g. 2008-09-15T05:43:00Z)
  url        the article on theguardian.com
  section    Guardian section name
  kind       "article" or "liveblog-block"
  last_modified / available_at   the API serves the latest version of each item, so search
             must filter on available_at (= last edit), not published

Leakage guards
  * Live blogs are split into their blocks, each stamped with the block's own first
    publication time. A whole live blog would carry updates written hours later.
  * Articles use firstPublicationDate, not webPublicationDate (which moves on re-publish).
  * Articles edited after publication can still contain later text (corrections,
    "This article was amended on ..."): lines like that are dropped, and the rest is a
    stated limitation.
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

EVENTS = {
    "lehman-2008": {
        "from": "2008-09-08",
        "to": "2008-09-19",
        "queries": [
            'Lehman',
            '"Merrill Lynch" OR "Bank of America"',
            'AIG OR "American International Group"',
            '"Wall Street" OR Paulson OR "Federal Reserve"',
            'Barclays OR HBOS OR "credit crunch"',
        ],
    },
    "titan-2023": {
        "from": "2023-06-18",
        "to": "2023-06-25",
        "queries": [
            'OceanGate OR submersible OR "Titan sub"',
            'Titanic AND (sub OR submersible OR missing OR search)',
            '"Stockton Rush" OR "Hamish Harding" OR "Shahzada Dawood" OR Nargeolet',
        ],
    },
}

API = "https://content.guardianapis.com/search"
AMENDED = re.compile(r"(this article was amended|this article was corrected|\bcorrection\b.*\d{4})", re.I)


def _get(client: httpx.Client, params: dict) -> dict:
    for attempt in range(5):
        r = client.get(API, params=params, timeout=60)
        if r.status_code == 429:
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        time.sleep(0.15)
        return r.json()["response"]
    raise RuntimeError("Guardian API kept returning 429")


def _clean(text: str) -> str:
    lines = [ln for ln in re.split(r"(?<=[.!?])\s+", text or "") if not AMENDED.search(ln)]
    return " ".join(lines).strip()


def _iso(ts: str | None) -> str:
    # Guardian returns e.g. 2008-09-15T05:43:00Z; keep it, normalise milliseconds away.
    if not ts:
        return ""
    return re.sub(r"\.\d+Z$", "Z", ts)


def fetch_event(name: str, key: str) -> pd.DataFrame:
    spec = EVENTS[name]
    seen: dict[str, dict] = {}
    with httpx.Client() as client:
        for q in spec["queries"]:
            page = 1
            while True:
                resp = _get(client, {
                    "q": q,
                    "from-date": spec["from"],
                    "to-date": spec["to"],
                    "page-size": 200,
                    "page": page,
                    "show-fields": "headline,bodyText,firstPublicationDate,lastModified,publication,byline",
                    "show-blocks": "body",
                    "api-key": key,
                })
                for item in resp["results"]:
                    seen.setdefault(item["id"], item)
                print(f"  {name}: q={q!r} page {page}/{resp['pages']} total={resp['total']}")
                if page >= resp["pages"]:
                    break
                page += 1

    rows = []
    for item in seen.values():
        f = item.get("fields", {})
        paper = f.get("publication") or "The Guardian"
        base = {
            "lccn": "",
            "paper": paper,
            "edition": "",
            "page": "",
            "url": item.get("webUrl", ""),
            "section": item.get("sectionName", ""),
        }
        if item.get("type") == "liveblog":
            for b in item.get("blocks", {}).get("body", []) or []:
                ts = _iso(b.get("firstPublishedDate") or b.get("publishedDate"))
                text = _clean(b.get("bodyTextSummary", ""))
                if not ts or not text:
                    continue
                rows.append({
                    **base,
                    "article_id": f"guardian:{item['id']}#{b['id']}",
                    "date": ts[:10],
                    "published": ts,
                    "last_modified": max(ts, _iso(b.get("lastModifiedDate")) or ts),
                    # Never fall back to the live blog's headline: it is the FINAL headline,
                    # rewritten as the story moved on (a hindsight leak).
                    "headline": (b.get("title") or "").strip(),
                    "text": text,
                    "url": f"{item.get('webUrl', '')}?page=with:block-{b['id']}#block-{b['id']}",
                    "kind": "liveblog-block",
                })
        else:
            ts = _iso(f.get("firstPublicationDate") or item.get("webPublicationDate"))
            text = _clean(f.get("bodyText", ""))
            if not text:
                continue
            rows.append({
                **base,
                "article_id": f"guardian:{item['id']}",
                "date": ts[:10],
                "published": ts,
                "last_modified": max(ts, _iso(f.get("lastModified")) or ts),
                "headline": (f.get("headline") or item.get("webTitle") or "").strip(),
                "text": text,
                "kind": "article",
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[(df["date"] >= spec["from"]) & (df["date"] <= spec["to"])]
    df = df.sort_values("published").drop_duplicates("article_id").reset_index(drop=True)
    # available_at: when the text we hold existed. The API serves the latest version, so an
    # article edited later only counts from its last edit (conservative: no hindsight leaks).
    # Edits more than 30 days later are archive housekeeping (the 2008 archive was re-saved
    # en masse), not news updates, so those items keep their publication time.
    lag = pd.to_datetime(df["last_modified"]) - pd.to_datetime(df["published"])
    housekeeping = lag > pd.Timedelta(days=30)
    df["available_at"] = df["last_modified"].where(~housekeeping, df["published"])
    cols = ["article_id", "lccn", "paper", "date", "edition", "page", "headline", "text",
            "published", "last_modified", "available_at", "url", "section", "kind"]
    return df[cols]


def main() -> None:
    key = os.getenv("GUARDIAN_API_KEY", "").strip()
    if not key or key.startswith("paste"):
        sys.exit("Set GUARDIAN_API_KEY in .env")
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    names = list(EVENTS) if which == "all" else [which]
    for name in names:
        df = fetch_event(name, key)
        out = ROOT / "data" / f"articles_{name}.parquet"
        df.to_parquet(out, index=False)
        by_day = df.groupby("date").size().to_dict() if not df.empty else {}
        print(f"{name}: {len(df)} rows -> {out.name}; per day {by_day}")


if __name__ == "__main__":
    main()
