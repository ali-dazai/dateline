"""Run the three setups over the verified test set and judge every answer.

Setups (same answer model for all three):
  prompt_only  model told the date, no retrieval            (eval/baseline.py)
  rag          engine answer(..., guard=False)               (date-locked search, no quote check)
  rag_guard    engine answer(..., guard=True)                (full Dateline)

The engine caches its model answer before the quote check, so `rag` and `rag_guard` judge the same
model answer for each row, with the check off and on.

Ablations (run only when named with --setups; rag_guard with one pipeline step turned off):
  rag_guard_no_expansion   DATELINE_TERM_EXPANSION=0: question words and story terms only, no terms call
  rag_guard_no_rerank      DATELINE_RERANK=0: the top 10 search hits in BM25 order, no re-rank call

Resumable: rows already in results/results.csv are skipped, so it can run in chunks
under free-tier rate limits. Every LLM call is cached on disk.

Usage:
  python -m eval.run_experiment                         # everything
  python -m eval.run_experiment --limit 20              # 20 new (question, cutoff, setup) runs, then stop
  python -m eval.run_experiment --setups prompt_only    # one setup only
  python -m eval.run_experiment --setups rag_guard_no_expansion
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from dateline import config
from dateline.contract import answer_text, validate_result
from dateline.engine import get_answer
from eval import baseline
from eval.judges import accuracy_judge, leakage_judge
from eval.testset import load, split_list

SETUPS = ["prompt_only", "rag", "rag_guard"]
# The pipeline steps an ablation can turn off, all on for the main setups: every retrieval setup sets all of
# them explicitly, so a stray environment variable cannot turn a main setup into an ablation.
SWITCHES = {"DATELINE_TERM_EXPANSION": "1", "DATELINE_RERANK": "1"}
# Ablation setup -> the environment settings that turn a pipeline step off (the engine reads them per answer
# and keys its answer cache on them, so ablation answers never mix with the normal ones).
ABLATIONS = {
    "rag_guard_no_expansion": {"DATELINE_TERM_EXPANSION": "0"},
    "rag_guard_no_rerank": {"DATELINE_RERANK": "0"},
}
ALL_SETUPS = SETUPS + list(ABLATIONS)
GUARDED = {"rag_guard", *ABLATIONS}
RESULT_COLUMNS = [
    "question_id", "cutoff", "setup", "answer_model", "judge_model", "engine",
    "answer", "quotes", "n_sentences", "n_removed", "contract_problems",
    "leaked", "facts_found", "leak_evidence", "accurate", "refused", "accuracy_reason",
    "run_at", "steps",
]


@contextlib.contextmanager
def _settings(values: dict[str, str]):
    """Set environment variables for the duration of one answer, then restore them."""
    before = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in before.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def run_one(row: pd.Series, setup: str) -> dict:
    cutoff = date.fromisoformat(row["cutoff"])
    question = row["question"]
    record: dict = {"question_id": row["question_id"], "cutoff": row["cutoff"], "setup": setup,
                    "answer_model": config.ANSWER_MODEL, "judge_model": config.JUDGE_MODEL,
                    "engine": "llm-only" if setup == "prompt_only" else config.ENGINE}

    if setup == "prompt_only":
        text = baseline.answer(question, cutoff)
        record.update(answer=text, quotes=json.dumps(baseline.extract_quotes(text)),
                      n_sentences="", n_removed="", contract_problems="", steps="")
    else:
        steps = {**SWITCHES, **ABLATIONS.get(setup, {})}
        with _settings(steps):
            result = get_answer()(question, cutoff, setup in GUARDED)
        text = answer_text(result)
        record.update(answer=text, steps=json.dumps(steps, sort_keys=True),
                      quotes=json.dumps([s["quote"] for s in result["sentences"]]),
                      n_sentences=len(result["sentences"]), n_removed=len(result["removed"]),
                      contract_problems=" | ".join(validate_result(result, cutoff)))

    leak = leakage_judge(question, cutoff, text, split_list(row["future_facts"]), row["period_answer"])
    acc = accuracy_judge(question, cutoff, text, row["period_answer"])
    record.update(leaked=leak["leaked"], facts_found="; ".join(leak["facts_found"]),
                  leak_evidence=leak["evidence"], accurate=acc["accurate"],
                  refused=acc["refused"], accuracy_reason=acc["reason"],
                  run_at=datetime.now().isoformat(timespec="seconds"))
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", default=str(config.DATA_DIR / "testset.csv"))
    parser.add_argument("--out", default=str(config.RESULTS_DIR / "results.csv"))
    parser.add_argument("--setups", nargs="+", default=SETUPS, choices=ALL_SETUPS)
    parser.add_argument("--limit", type=int, default=0, help="stop after N new runs (0 = no limit)")
    args = parser.parse_args()

    tests = load(args.testset)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        prev = pd.read_csv(out, dtype=str, keep_default_na=False)
        missing = [c for c in RESULT_COLUMNS if c not in prev.columns]
        if missing:  # a file from before a column was added: add it, empty, so appended rows line up
            for c in missing:
                prev[c] = ""
            prev[RESULT_COLUMNS + [c for c in prev.columns if c not in RESULT_COLUMNS]].to_csv(out, index=False)
        done = set(zip(prev["question_id"], prev["cutoff"], prev["setup"]))

    todo = [(row, s) for _, row in tests.iterrows() for s in args.setups
            if (row["question_id"], row["cutoff"], s) not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(tests)} verified test rows, {len(done)} runs already done, {len(todo)} to run now.")

    for n, (row, setup) in enumerate(todo, 1):
        record = run_one(row, setup)
        pd.DataFrame([record], columns=RESULT_COLUMNS).to_csv(
            out, mode="a", header=not out.exists(), index=False)
        print(f"[{n}/{len(todo)}] {row['question_id']} @ {row['cutoff']} {setup}: "
              f"leaked={record['leaked']} accurate={record['accurate']}")


if __name__ == "__main__":
    main()
