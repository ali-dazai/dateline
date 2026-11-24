"""Real-data check of the retrieval pipeline: contract, removals, flags and latency.

For each row the pipeline answers three times through `pipeline.engine.answer(..., interactive=True)`:
guard on (a new answer), guard off (reuses the cached model answer) and guard on again (times a cache
hit). Writes one CSV row per (question, guard) with sentence and removal counts, removal reasons, flags,
notice, contract problems (`validate_result`), seconds (new and cached) and seconds per step, plus a
JSON Lines file with every removed or flagged sentence and the passage it cites, for reading by hand.

Sets:
  tuning  the first 30 rows of data/preset_candidates.csv (P01-P30) and the first 15 rows of data/testset.csv
  full    all 30 rows of data/testset.csv

    python -u scripts/check_pipeline.py --set tuning
    python -u scripts/check_pipeline.py --set full --fresh

`--fresh` bypasses the model-response cache and keeps answers in a new folder under cache/, so every
"new" time is a real run (model calls are paid again).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dateline import config, events, llm  # noqa: E402
from dateline.contract import validate_result  # noqa: E402
from pipeline import engine  # noqa: E402
from pipeline import index as _index  # noqa: E402

TUNING_CANDIDATES = 30
STEPS = ("terms", "search", "rerank", "passages", "compose", "classify", "checks")
COLUMNS = ["event", "row_id", "cutoff", "question", "guard", "n_sentences", "n_removed", "removed_reasons",
           "flags", "notice", "problems", "seconds_new", "seconds_cached", "reused_model_answer",
           *[f"s_{step}" for step in STEPS]]


class _Steps(logging.Handler):
    """Collects the engine's per-step duration records."""

    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.steps: dict[str, float] = {}

    def emit(self, record: logging.LogRecord) -> None:
        step = getattr(record, "step", None)
        if step:
            self.steps[step] = self.steps.get(step, 0.0) + float(getattr(record, "seconds", 0.0))


def _rows(which: str) -> list[dict]:
    with (ROOT / "data" / "testset.csv").open(encoding="utf-8") as fh:
        test = list(csv.DictReader(fh))
    test_rows = [{"event": events.DEFAULT_EVENT, "row_id": f"{r['question_id']}@{r['cutoff']}",
                  "cutoff": r["cutoff"], "question": r["question"]} for r in test]
    if which == "full":
        return test_rows
    # The tuning set is pinned (docs/pipeline-design.md, "Real-data check and tuning"): the 30 candidates
    # the constants and prompts were tuned on, P01-P30. Rows added to the candidates file later are not
    # part of it.
    with (ROOT / "data" / "preset_candidates.csv").open(encoding="utf-8") as fh:
        presets = [{"event": r["event"], "row_id": f"P{i:02d}", "cutoff": r["cutoff"], "question": r["question"]}
                   for i, r in enumerate(csv.DictReader(fh), 1)][:TUNING_CANDIDATES]
    return presets + test_rows[:15]


def _call(handler: _Steps, row: dict, guard: bool) -> tuple[dict, float, dict[str, float], str]:
    handler.steps = {}
    started = time.perf_counter()
    try:
        result = engine.answer(row["question"], row["cutoff"], guard, row["event"], interactive=True)
        error = ""
    except Exception as exc:  # noqa: BLE001 - a crash is a finding, the run goes on
        result, error = {"sentences": [], "removed": [], "groups": [], "retrieved": []}, f"exception: {exc!r}"
    return result, time.perf_counter() - started, dict(handler.steps), error


def _state(row: dict) -> dict | None:
    """The cached pre-check state of the row's answer (passages and draft), if stored."""
    try:
        limit = events.parse_cutoff(row["event"], row["cutoff"])
        con = _index.connect(row["event"])
        try:
            rowid = _index.max_rowid(con, limit)
        finally:
            con.close()
        when = engine.moment(row["event"], limit)
        return engine._load(engine._cache_key(row["event"], row["question"], rowid, when, engine.switches()))
    except Exception:  # noqa: BLE001
        return None


def _pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = max(0, math.ceil(q * len(ordered)) - 1)
    return ordered[k]


def _fresh() -> Path:
    """Bypass the model-response cache and keep answers in a new folder, so every run is new."""
    run = ROOT / "cache" / f"check_{datetime.now():%Y%m%d_%H%M%S}"
    original = llm.complete

    def uncached(*args, **kwargs):
        kwargs["use_cache"] = False
        return original(*args, **kwargs)

    llm.complete = uncached
    engine._cache_path = lambda key: run / "answers" / f"{key}.json"
    return run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", choices=("tuning", "full"), default="tuning")
    parser.add_argument("--fresh", action="store_true", help="bypass the model-response and answer caches")
    parser.add_argument("--only", default="", help="comma-separated row ids to run")
    parser.add_argument("--tag", default="", help="suffix for the output file names")
    args = parser.parse_args()

    if args.fresh:
        print(f"fresh run: answers in {_fresh()}", flush=True)
    handler = _Steps()
    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    rows = _rows(args.set)
    if args.only:
        wanted = set(args.only.split(","))
        rows = [r for r in rows if r["row_id"] in wanted]
    name = f"pipeline_check_{args.set}{('_' + args.tag) if args.tag else ''}"
    out_csv = config.RESULTS_DIR / f"{name}.csv"
    out_jsonl = config.RESULTS_DIR / f"{name}.jsonl"
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{len(rows)} rows, model {config.ANSWER_MODEL}", flush=True)

    table: list[dict] = []
    new_times: list[float] = []
    cached_times: list[float] = []
    step_times: dict[str, list[float]] = {s: [] for s in STEPS}
    with out_jsonl.open("w", encoding="utf-8") as detail:
        for n, row in enumerate(rows, 1):
            on, s_new, steps_new, err_on = _call(handler, row, True)
            off, s_off, steps_off, err_off = _call(handler, row, False)
            again, s_cached, _steps, err_again = _call(handler, row, True)
            reused = set(steps_off) <= {"checks"}
            new_times.append(s_new)
            cached_times.append(s_cached)
            for step in STEPS:
                if step in steps_new:
                    step_times[step].append(steps_new[step])
            flags = [s for s in off["sentences"] if s.get("flag")]
            for guard, result, seconds, err in ((True, on, s_new, err_on), (False, off, s_off, err_off)):
                problems = validate_result(result, row["cutoff"]) + ([err] if err else [])
                if guard and err_again:
                    problems.append(f"cached call: {err_again}")
                table.append({
                    "event": row["event"], "row_id": row["row_id"], "cutoff": row["cutoff"],
                    "question": row["question"], "guard": guard,
                    "n_sentences": len(result["sentences"]), "n_removed": len(result["removed"]),
                    "removed_reasons": " | ".join(r["reason"] for r in result["removed"]),
                    "flags": " | ".join(s["flag"] for s in result["sentences"] if s.get("flag")),
                    "notice": result.get("notice", ""), "problems": " | ".join(problems),
                    "seconds_new": round(seconds, 3) if guard else "",
                    "seconds_cached": round(s_cached, 4) if guard else round(s_off, 4),
                    "reused_model_answer": "" if guard else reused,
                    **{f"s_{step}": round(steps_new[step], 3) if guard and step in steps_new else ""
                       for step in STEPS},
                })
            state = _state(row) or {}
            passages = state.get("passages", {})
            detail.write(json.dumps({
                **row,
                "notice": on.get("notice", ""),
                "kept": [{"text": s["text"], "quote": s["quote"], "article_id": s["article_id"],
                          "paper": s["paper"]} for s in on["sentences"]],
                "flagged": [{"text": s["text"], "quote": s["quote"], "article_id": s["article_id"],
                             "paper": s["paper"], "flag": s["flag"], "match": s.get("match"),
                             "passage": passages.get(s["article_id"], "")} for s in flags],
                "removed": on["removed"],
                "groups": [{"claim": g["claim"], "count": g["count"]} for g in on["groups"]],
                "chart": bool(on.get("chart")),
            }, ensure_ascii=False) + "\n")
            print(f"[{n}/{len(rows)}] {row['row_id']} {row['event']} {row['cutoff']} | "
                  f"{len(on['sentences'])} kept, {len(on['removed'])} removed | new {s_new:.1f}s, "
                  f"cached {s_cached * 1000:.0f}ms | {on.get('notice', '')} "
                  f"{'| PROBLEMS' if validate_result(on, row['cutoff']) or err_on else ''}", flush=True)

    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(table)

    problems = sum(1 for r in table if r["problems"])
    empty = sum(1 for r in table if r["guard"] and r["n_sentences"] == 0 and not r["notice"])
    print(f"\nwrote {out_csv} and {out_jsonl}")
    print(f"rows with contract problems: {problems}; rows with no sentence and no notice: {empty}")
    print(f"new answer: p50 {_pct(new_times, 0.5):.2f} s, p95 {_pct(new_times, 0.95):.2f} s, "
          f"max {max(new_times, default=float('nan')):.2f} s")
    print(f"cached answer: p50 {_pct(cached_times, 0.5) * 1000:.0f} ms, p95 {_pct(cached_times, 0.95) * 1000:.0f} ms, "
          f"max {max(cached_times, default=float('nan')) * 1000:.0f} ms")
    print("step means (new answers): " + ", ".join(
        f"{s} {statistics.mean(v):.2f} s (n={len(v)})" for s, v in step_times.items() if v))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
