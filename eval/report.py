"""Summarise results/results.csv into the headline numbers and a plain chart.

Leakage rate is computed only over questions the model has memorised (memorised == yes);
otherwise "no leak" could just mean "the model didn't know". An empty answer cannot leak, so a
setup that removes everything would look leak-free: each setup therefore also reports the share of
answers that are empty or fully removed, and leakage over non-empty answers only.

The pipeline was tuned on the first 15 test-set rows (Q01-Q07); the retrieval setups are also
reported separately for those tuning rows and the 15 held-out rows (Q08-Q15).

Usage:  python -m eval.report
Writes results/summary.csv, results/summary_split.csv, results/summary.md and results/chart.png.
"""
from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from dateline import config  # noqa: E402
from eval.run_experiment import ALL_SETUPS  # noqa: E402

LABELS = {"prompt_only": "Prompt only", "rag": "RAG", "rag_guard": "RAG + quote check",
          "rag_guard_no_expansion": "RAG + quote check, no term expansion",
          "rag_guard_no_rerank": "RAG + quote check, no re-rank"}
TUNING_QUESTIONS = {f"Q{i:02d}" for i in range(1, 8)}  # the first 15 test-set rows


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def _column(r: pd.DataFrame, name: str) -> pd.Series:
    return r[name].fillna("").astype(str) if name in r.columns else pd.Series("", index=r.index)


def _rate(k: int, n: int) -> float:
    return k / n if n else float("nan")


def _measures(r: pd.DataFrame, memorised: pd.DataFrame) -> dict:
    """The numbers for one group of runs (one setup, or one setup on part of the rows)."""
    empty = _column(r, "answer").str.strip() == ""
    removed = pd.to_numeric(_column(r, "n_removed"), errors="coerce").fillna(0)
    all_removed = empty & (removed > 0)
    leaked = _column(r, "leaked").str.lower() == "true"
    on_memorised = r.merge(memorised, on=["question_id", "cutoff"], how="left", indicator=True)["_merge"] == "both"
    on_memorised.index = r.index
    pool = on_memorised
    pool_full = pool & ~empty
    leaks, leaks_full = int((leaked & pool).sum()), int((leaked & pool_full).sum())
    n, n_full = int(pool.sum()), int(pool_full.sum())
    grade = _column(r, "accurate")
    acc_yes = int((grade == "yes").sum())
    lo, hi = wilson(leaks, n)
    lo_f, hi_f = wilson(leaks_full, n_full)
    a_lo, a_hi = wilson(acc_yes, len(r))
    return {
        "runs": len(r),
        "leaks": leaks, "leakage_n": n, "leakage_rate": _rate(leaks, n),
        "leakage_ci_low": lo, "leakage_ci_high": hi,
        "leaks_nonempty": leaks_full, "leakage_nonempty_n": n_full,
        "leakage_nonempty_rate": _rate(leaks_full, n_full),
        "leakage_nonempty_ci_low": lo_f, "leakage_nonempty_ci_high": hi_f,
        "accuracy_yes_n": acc_yes, "accuracy_partial_n": int((grade == "partial").sum()),
        "accuracy_no_n": int((grade == "no").sum()),
        "accuracy_yes": _rate(acc_yes, len(r)), "accuracy_ci_low": a_lo, "accuracy_ci_high": a_hi,
        "accuracy_partial": _rate(int((grade == "partial").sum()), len(r)),
        "empty_or_removed_n": int(empty.sum()), "empty_or_removed": _rate(int(empty.sum()), len(r)),
        "all_removed_n": int(all_removed.sum()),
        "refusal_rate": _rate(int((_column(r, "refused").str.lower() == "true").sum()), len(r)),
    }


def _memorised(tests: pd.DataFrame) -> pd.DataFrame:
    return tests[tests["memorised"].str.lower() == "yes"][["question_id", "cutoff"]]


def summarise(results: pd.DataFrame, tests: pd.DataFrame) -> pd.DataFrame:
    memorised = _memorised(tests)
    rows = []
    for setup in ALL_SETUPS:
        r = results[results["setup"] == setup]
        if r.empty:
            continue
        rows.append({"setup": LABELS[setup], **_measures(r, memorised)})
    return pd.DataFrame(rows)


def split_summary(results: pd.DataFrame, tests: pd.DataFrame) -> pd.DataFrame:
    """The retrieval setups on the tuning rows (Q01-Q07) and the held-out rows (Q08-Q15) separately."""
    memorised = _memorised(tests)
    tuning = results["question_id"].isin(TUNING_QUESTIONS)
    rows = []
    for setup in ALL_SETUPS:
        if setup == "prompt_only":
            continue
        for part, mask in (("tuning (Q01-Q07)", tuning), ("held-out (Q08-Q15)", ~tuning)):
            r = results[(results["setup"] == setup) & mask]
            if r.empty:
                continue
            rows.append({"setup": LABELS[setup], "rows": part, **_measures(r, memorised)})
    return pd.DataFrame(rows)


def _pct(x: float) -> str:
    return "n/a" if x != x else f"{100 * x:.0f}%"


def _ci(lo: float, hi: float) -> str:
    return "n/a" if lo != lo else f"{100 * lo:.0f}–{100 * hi:.0f}%"


def _table(df: pd.DataFrame, keys: dict[str, str]) -> list[str]:
    head = [*keys, "Leakage", "95% CI", "Leakage, non-empty answers", "95% CI",
            "Accurate (yes / partial / no)", "Accuracy yes, 95% CI", "Empty or all removed"]
    lines = ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    for _, s in df.iterrows():
        cells = [str(s[col]) for col in keys.values()]
        cells += [
            f"{s['leaks']}/{s['leakage_n']} ({_pct(s['leakage_rate'])})",
            _ci(s["leakage_ci_low"], s["leakage_ci_high"]),
            f"{s['leaks_nonempty']}/{s['leakage_nonempty_n']} ({_pct(s['leakage_nonempty_rate'])})",
            _ci(s["leakage_nonempty_ci_low"], s["leakage_nonempty_ci_high"]),
            f"{s['accuracy_yes_n']} / {s['accuracy_partial_n']} / {s['accuracy_no_n']}",
            f"{_pct(s['accuracy_yes'])} ({_ci(s['accuracy_ci_low'], s['accuracy_ci_high'])})",
            f"{s['empty_or_removed_n']}/{s['runs']} ({_pct(s['empty_or_removed'])}; "
            f"{s['all_removed_n']} all removed)",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def markdown(summary: pd.DataFrame, split: pd.DataFrame) -> str:
    out = ["## All rows", "", *_table(summary, {"Setup": "setup"})]
    if not split.empty:
        out += ["", "## Tuning rows and held-out rows", "", *_table(split, {"Setup": "setup", "Rows": "rows"})]
    return "\n".join(out) + "\n"


def chart(summary: pd.DataFrame, path) -> None:
    fig, ax = plt.subplots(figsize=(max(7, 1.9 * len(summary)), 4.2))
    x = list(range(len(summary)))
    w = 0.38
    leak = summary["leakage_rate"] * 100
    acc = summary["accuracy_yes"] * 100
    leak_err = [leak - summary["leakage_ci_low"] * 100, summary["leakage_ci_high"] * 100 - leak]
    acc_err = [acc - summary["accuracy_ci_low"] * 100, summary["accuracy_ci_high"] * 100 - acc]
    ax.bar([i - w / 2 for i in x], leak, w, yerr=leak_err, capsize=3, label="Hindsight leakage (%)")
    ax.bar([i + w / 2 for i in x], acc, w, yerr=acc_err, capsize=3, label="Period accuracy, yes (%)")
    ax.set_xticks(x, [s.replace(", ", ",\n") for s in summary["setup"]], fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of answers (95% CI)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def main() -> None:
    results = pd.read_csv(config.RESULTS_DIR / "results.csv", dtype=str).fillna("")
    tests = pd.read_csv(config.DATA_DIR / "testset.csv", dtype=str).fillna("")
    summary = summarise(results, tests)
    split = split_summary(results, tests)
    summary.to_csv(config.RESULTS_DIR / "summary.csv", index=False)
    split.to_csv(config.RESULTS_DIR / "summary_split.csv", index=False)
    (config.RESULTS_DIR / "summary.md").write_text(markdown(summary, split), encoding="utf-8")
    chart(summary, config.RESULTS_DIR / "chart.png")
    print(markdown(summary, split))


if __name__ == "__main__":
    main()
