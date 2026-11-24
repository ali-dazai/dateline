"""Build the article table from an American Stories year file.

Streams data/raw/faro_<year>.tar.gz (one JSON per newspaper page scan) and keeps
articles whose issue date falls in [--start, --end]. Writes a parquet with the
columns the contract expects:
  article_id, lccn, paper, state, date, edition, page, headline, byline, text, url, scan_url, event

Usage:
  python scripts/build_articles.py --year 1912 --start 1912-04-10 --end 1912-04-20 --event titanic
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tarfile
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
NAME = re.compile(r"(\d{4}-\d{2}-\d{2})_p([^_]+)_(sn\w+?)_")


def clean(text: str) -> str:
    text = (text or "").replace("-\n", "")
    text = re.sub(r"\s*\n\s*", " ", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def paper_name(lccn_meta: dict) -> str:
    title = lccn_meta.get("title", "")
    return re.sub(r"\s*\[volume\]\s*", " ", title).strip().rstrip(".")


def rows_from_scan(scan_id: str, page: str, d: dict, event: str) -> list[dict]:
    meta = d.get("lccn", {}) or {}
    lccn = meta.get("lccn") or d.get("scan", {}).get("lccn", "")
    edition_raw = (d.get("edition", {}) or {}).get("edition", "ed-01")
    edition = str(int(re.sub(r"\D", "", edition_raw) or 1))
    date = (d.get("edition", {}) or {}).get("date") or d.get("scan", {}).get("date", "")
    url = f"https://www.loc.gov/item/{lccn}/{date}/ed-{edition}/"
    scan_url = (d.get("scan", {}) or {}).get("raw_data_loc", "")
    rows = []
    for art in d.get("full articles", []) or []:
        text = clean(art.get("article", ""))
        if len(text) < 40:
            continue
        rows.append({
            "article_id": f"{scan_id}_{art.get('full_article_id', art.get('id'))}",
            "lccn": lccn,
            "paper": paper_name(meta),
            "state": meta.get("state", ""),
            "date": date,
            "edition": edition,
            "page": page,
            "headline": clean(art.get("headline", "")),
            "byline": clean(art.get("byline", "")),
            "text": text,
            "url": url,
            "scan_url": scan_url,
            "event": event,
        })
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--year", required=True)
    p.add_argument("--start", required=True, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--end", required=True, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--event", default="")
    p.add_argument("--out", default="")
    args = p.parse_args()

    src = ROOT / "data" / "raw" / f"faro_{args.year}.tar.gz"
    out = Path(args.out) if args.out else ROOT / "data" / f"articles_{args.start}_{args.end}.parquet"
    rows: list[dict] = []
    scans = kept = 0
    t0 = time.time()
    with tarfile.open(src, "r|gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            scans += 1
            name = member.name.rsplit("/", 1)[-1]
            m = NAME.match(name)
            if not m or not (args.start <= m.group(1) <= args.end):
                continue
            d = json.load(tar.extractfile(member))
            rows.extend(rows_from_scan(name[:-5], m.group(2), d, args.event))
            kept += 1
            if kept % 2000 == 0:
                print(f"{kept} scans kept / {scans} read, {len(rows)} articles, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)
    print(f"Done: {kept} scans, {len(df)} articles, {df['lccn'].nunique() if len(df) else 0} papers -> {out} "
          f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    sys.exit(main())
