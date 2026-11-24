"""Memorisation check: does the model already know the outcome of each question?

Leakage can only be measured for questions where the model knows what happened next.
For each question we ask it with NO cutoff, then reuse the leakage judge: if the free
answer contains a future fact, the model has memorised the outcome.

Usage:  python -m eval.memorisation data/testset.csv     (fills the `memorised` column)
"""
from __future__ import annotations

import sys
from datetime import date

import pandas as pd

from dateline import config, llm
from eval.judges import leakage_judge
from eval.testset import COLUMNS, split_list

FREE_SYSTEM = "Answer from your own knowledge of history. Be specific about what eventually happened."


def check_row(question: str, future_facts: list[str]) -> bool:
    reply = llm.complete(question, system=FREE_SYSTEM, model=config.ANSWER_MODEL)
    # A far-future "cutoff" just phrases the judge prompt; what matters is the fact list.
    verdict = leakage_judge(question, date(1912, 4, 10), reply, future_facts)
    return verdict["leaked"]


def main(path: str) -> None:
    df = pd.read_csv(path, dtype=str).fillna("")
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = ""
    known: dict[str, bool] = {}
    for i, row in df.iterrows():
        qid = row["question_id"]
        if qid not in known:
            facts = split_list(row["future_facts"])
            known[qid] = check_row(row["question"], facts) if facts else False
            print(f"{qid}: {'memorised' if known[qid] else 'NOT memorised'}")
        df.at[i, "memorised"] = "yes" if known[qid] else "no"
    df[COLUMNS].to_csv(path, index=False)
    kept = sum(known.values())
    print(f"\n{kept}/{len(known)} questions memorised. Questions marked 'no' are excluded from leakage results.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(config.DATA_DIR / "testset.csv"))
