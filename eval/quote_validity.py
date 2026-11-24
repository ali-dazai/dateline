"""Quote validity for the prompt-only baseline: do the newspaper quotes it gives really exist?

For each quoted passage in a prompt_only answer, look for it in articles dated on or
before the cutoff (fuzzy match, to tolerate OCR typos). Needs data/articles.parquet.

Usage:  python -m eval.quote_validity [--corpus data/articles.parquet] [--threshold 85]
Writes results/quote_validity.csv and prints the share of quotes found.
"""
from __future__ import annotations

import argparse
import json
import re

import pandas as pd
from rapidfuzz import fuzz

from dateline import config

_WORD = re.compile(r"[a-z]{4,}")


def _anchor_words(quote: str, k: int = 2) -> list[str]:
    words = sorted(set(_WORD.findall(quote.lower())), key=len, reverse=True)
    return words[:k]


def find_quote(quote: str, articles: pd.DataFrame, threshold: int) -> tuple[bool, str, int]:
    """Return (found, article_id, best score). Prefilters on the quote's longest words."""
    candidates = articles
    for word in _anchor_words(quote):
        candidates = candidates[candidates["text_lower"].str.contains(word, regex=False)]
        if candidates.empty:
            return False, "", 0
    best_id, best = "", 0
    for art_id, text in zip(candidates["article_id"], candidates["text_lower"]):
        score = fuzz.partial_ratio(quote.lower(), text)
        if score > best:
            best_id, best = art_id, score
            if score >= 98:
                break
    return best >= threshold, best_id, int(best)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(config.RESULTS_DIR / "results.csv"))
    parser.add_argument("--corpus", default=config.CORPUS_PATH)
    parser.add_argument("--threshold", type=int, default=85)
    args = parser.parse_args()

    corpus = pd.read_parquet(args.corpus, columns=["article_id", "date", "text"])
    corpus["text_lower"] = corpus["text"].fillna("").str.lower()
    results = pd.read_csv(args.results, dtype=str).fillna("")
    rows = []
    for _, r in results[results["setup"] == "prompt_only"].iterrows():
        allowed = corpus[corpus["date"].astype(str) <= r["cutoff"]]
        for quote in json.loads(r["quotes"] or "[]"):
            found, art_id, score = find_quote(quote, allowed, args.threshold)
            rows.append({"question_id": r["question_id"], "cutoff": r["cutoff"], "quote": quote,
                         "found": found, "best_article_id": art_id, "score": score})
    out = pd.DataFrame(rows)
    path = config.RESULTS_DIR / "quote_validity.csv"
    out.to_csv(path, index=False)
    if out.empty:
        print("The prompt-only answers contained no quotes.")
    else:
        print(f"{out['found'].mean():.0%} of {len(out)} quoted passages were found in the archive. -> {path}")


if __name__ == "__main__":
    main()
