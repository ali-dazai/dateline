"""Load and check the hand-verified test set (data/testset.csv).

One row = one question asked at one cutoff date. Columns:
  question_id        e.g. Q01 (same id for the same question at different cutoffs)
  question           the question as a user would ask it
  cutoff             YYYY-MM-DD; the system may only use articles dated on or before this
  period_answer      what papers dated <= cutoff said (written/checked by a human)
  future_facts       facts only known AFTER the cutoff, separated by ";"
  source_article_ids articles that support period_answer, separated by ";"
  source_urls        links to those articles/pages, separated by ";"
  memorised          yes/no: does the model know the outcome? (filled by memorisation.py)
  verified_by        who checked period_answer against the articles (a person, or the
                     quote check in scripts/build_testset.py)
  notes              free text

Rows without verified_by are unverified and are refused by the experiment. Every period
answer must be grounded in period articles, never in a model's knowledge of the outcome;
otherwise the leakage measurement would be circular.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

COLUMNS = [
    "question_id", "question", "cutoff", "period_answer", "future_facts",
    "source_article_ids", "source_urls", "memorised", "verified_by", "notes",
]


def split_list(value: object) -> list[str]:
    if not isinstance(value, str):
        return []
    return [part.strip() for part in value.split(";") if part.strip()]


def load(path: str | Path, *, verified_only: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"testset is missing columns: {missing}")
    problems = check(df)
    if problems:
        raise ValueError("testset problems:\n" + "\n".join(problems))
    if verified_only:
        df = df[df["verified_by"].str.strip() != ""]
    return df.reset_index(drop=True)


def check(df: pd.DataFrame) -> list[str]:
    problems = []
    for i, row in df.iterrows():
        where = f"row {i + 2} ({row['question_id']} @ {row['cutoff']})"
        try:
            date.fromisoformat(row["cutoff"])
        except ValueError:
            problems.append(f"{where}: cutoff must be YYYY-MM-DD")
        for col in ("question_id", "question", "period_answer"):
            if not row[col].strip():
                problems.append(f"{where}: {col} is empty")
        if not split_list(row["future_facts"]):
            problems.append(f"{where}: no future_facts, so leakage can't be judged")
    dupes = df[df.duplicated(["question_id", "cutoff"], keep=False)]
    for _, row in dupes.iterrows():
        problems.append(f"duplicate question_id + cutoff: {row['question_id']} @ {row['cutoff']}")
    return problems


def cutoff_phrase(cutoff: date) -> str:
    """'the end of Monday, April 15, 1912' (the cutoff day itself is included)."""
    return f"the end of {cutoff.strftime('%A')}, {cutoff.strftime('%B')} {cutoff.day}, {cutoff.year}"
