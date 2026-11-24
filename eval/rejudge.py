"""Re-run the judges on answers already in results/results.csv, without new answers.

Use after a judge prompt changes. Answers are reused as stored, so only judge calls are
made (each cached on disk). The previous file is kept as results/results_before_rejudge.csv.

Usage:
  python -m eval.rejudge                          # leakage judge, all setups
  python -m eval.rejudge --judges leakage accuracy --setups prompt_only
"""
from __future__ import annotations

import argparse
import shutil
from datetime import date
from pathlib import Path

import pandas as pd

from dateline import config
from eval.judges import accuracy_judge, leakage_judge
from eval.testset import load, split_list


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(config.RESULTS_DIR / "results.csv"))
    parser.add_argument("--testset", default=str(config.DATA_DIR / "testset.csv"))
    parser.add_argument("--judges", nargs="+", default=["leakage"], choices=["leakage", "accuracy"])
    parser.add_argument("--setups", nargs="+", default=None, help="only these setups (default: all)")
    args = parser.parse_args()

    path = Path(args.results)
    results = pd.read_csv(path, dtype=str, keep_default_na=False)
    tests = load(args.testset).set_index(["question_id", "cutoff"])
    backup = path.with_name(path.stem + "_before_rejudge.csv")
    shutil.copyfile(path, backup)

    changed = 0
    for i, row in results.iterrows():
        if args.setups and row["setup"] not in args.setups:
            continue
        key = (row["question_id"], row["cutoff"])
        if key not in tests.index:
            print(f"skip {key}: not in the test set")
            continue
        test = tests.loc[key]
        cutoff = date.fromisoformat(row["cutoff"])
        before = (row["leaked"], row["accurate"])
        if "leakage" in args.judges:
            leak = leakage_judge(test["question"], cutoff, row["answer"],
                                 split_list(test["future_facts"]), test["period_answer"])
            results.loc[i, ["leaked", "facts_found", "leak_evidence"]] = [
                str(leak["leaked"]), "; ".join(leak["facts_found"]), leak["evidence"]]
        if "accuracy" in args.judges:
            acc = accuracy_judge(test["question"], cutoff, row["answer"], test["period_answer"])
            results.loc[i, ["accurate", "refused", "accuracy_reason"]] = [
                acc["accurate"], str(acc["refused"]), acc["reason"]]
        results.loc[i, "judge_model"] = config.JUDGE_MODEL
        after = (results.loc[i, "leaked"], results.loc[i, "accurate"])
        mark = "changed" if after != before else "same"
        changed += after != before
        print(f"{row['question_id']} @ {row['cutoff']} {row['setup']}: "
              f"leaked {before[0]} -> {after[0]}, accurate {before[1]} -> {after[1]} ({mark})", flush=True)

    results.to_csv(path, index=False)
    print(f"{changed} verdicts changed. Previous results kept in {backup}.")


if __name__ == "__main__":
    main()
