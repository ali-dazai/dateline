"""Score the preset candidates: what the plain model says and what Dateline says, each judged for hindsight
and accuracy against the row's answer key (data/preset_candidates.csv).

For each candidate: the plain answer (the prompt-only baseline told the moment, as `/api/plain` serves it)
and the pipeline answer (`pipeline.engine.answer`, quote check on, batch limits), both scored with
`eval.judges.leakage_judge` and `accuracy_judge` against the row's `period_answer` and `future_facts`.

    keep      = plain_bad and pipeline_good
    plain_bad = the plain answer leaked, or its accuracy is "no"
    pipeline_good = no leak, accuracy "yes", and at least 2 sentences
    near_miss = plain_bad and the pipeline's accuracy is "partial" (used only to fill a story short of presets)

Writes results/preset_selection.csv (both answers, verdicts, judge notes, counts, keep) and
results/preset_selection.md (a table to read). Rows without a period answer are answered but not judged.

    python -u scripts/select_presets.py [--only P01,P02]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dateline import config, events  # noqa: E402
from eval import baseline, judges  # noqa: E402
from eval.testset import split_list  # noqa: E402
from pipeline import engine  # noqa: E402

CANDIDATES = ROOT / "data" / "preset_candidates.csv"
COLUMNS = ["row_id", "event", "cutoff", "question", "plain_answer", "plain_leaked", "plain_accurate",
           "plain_note", "pipeline_answer", "pipeline_leaked", "pipeline_accurate", "pipeline_note",
           "n_sentences", "n_removed", "n_groups", "notice", "plain_bad", "pipeline_good", "keep", "near_miss"]
EXCERPT = 260


def decide(plain: dict, pipe: dict, n_sentences: int) -> dict:
    """The keep rule, from the two pairs of verdicts ({"leaked": bool, "accurate": str})."""
    plain_bad = bool(plain["leaked"]) or plain["accurate"] == "no"
    pipeline_good = not pipe["leaked"] and pipe["accurate"] == "yes" and n_sentences >= 2
    return {"plain_bad": plain_bad, "pipeline_good": pipeline_good, "keep": plain_bad and pipeline_good,
            "near_miss": plain_bad and pipe["accurate"] == "partial"}


def _judge(question: str, event: str, cutoff: str, text: str, period: str, future: list[str]) -> dict:
    at = events.parse_cutoff(event, cutoff)
    when = None if events.get(event)["granularity"] == "day" else engine.moment(event, at)
    leak = judges.leakage_judge(question, at.date(), text, future, period, when=when)
    acc = judges.accuracy_judge(question, at.date(), text, period, when=when)
    note = "; ".join(part for part in (
        f"leak: {leak['evidence']}" if leak.get("leaked") and leak.get("evidence") else "",
        f"facts: {', '.join(leak['facts_found'])}" if leak.get("facts_found") else "",
        f"accuracy: {acc['reason']}" if acc.get("reason") else "",
    ) if part)
    return {"leaked": leak["leaked"], "accurate": acc["accurate"], "note": note}


def _judge_safely(*args) -> dict:
    """A judge reply that cannot be read is recorded and the run goes on; such a row is never kept."""
    try:
        return _judge(*args)
    except Exception as exc:  # noqa: BLE001
        return {"leaked": True, "accurate": "no", "note": f"judge failed: {type(exc).__name__}"}


def rows(path: Path = CANDIDATES) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [{"row_id": f"P{i:02d}", **r} for i, r in enumerate(csv.DictReader(fh), 1)]


def score(row: dict) -> dict:
    question, event, cutoff = row["question"], row["event"], row["cutoff"]
    plain = baseline.answer_at(question, event, cutoff)
    result = engine.answer(question, cutoff, True, event, interactive=False)
    pipe_text = " ".join(s["text"] for s in result["sentences"])
    out = {"row_id": row["row_id"], "event": event, "cutoff": cutoff, "question": question,
           "plain_answer": plain, "pipeline_answer": pipe_text, "n_sentences": len(result["sentences"]),
           "n_removed": len(result["removed"]), "n_groups": len(result["groups"]),
           "notice": result.get("notice", "")}
    period, future = row.get("period_answer", "").strip(), split_list(row.get("future_facts", ""))
    if not period:
        return {**out, "plain_note": "no answer key", "pipeline_note": "no answer key",
                **dict.fromkeys(("plain_leaked", "plain_accurate", "pipeline_leaked", "pipeline_accurate",
                                 "plain_bad", "pipeline_good", "keep", "near_miss"), "")}
    p = _judge_safely(question, event, cutoff, plain, period, future)
    d = _judge_safely(question, event, cutoff, pipe_text, period, future)
    return {**out, "plain_leaked": p["leaked"], "plain_accurate": p["accurate"], "plain_note": p["note"],
            "pipeline_leaked": d["leaked"], "pipeline_accurate": d["accurate"], "pipeline_note": d["note"],
            **_decision(p, d, out["n_sentences"])}


def _decision(plain: dict, pipe: dict, n_sentences: int) -> dict:
    out = decide(plain, pipe, n_sentences)
    if any(v["note"].startswith("judge failed") for v in (plain, pipe)):
        out.update(keep=False, near_miss=False)
    return out


def _cell(text: object, limit: int | None = None) -> str:
    flat = " ".join(str(text or "").split()).replace("|", "\\|")
    return flat if limit is None or len(flat) <= limit else flat[:limit].rstrip() + "…"


def _verdict(leaked: object, accurate: object) -> str:
    if leaked == "" and accurate == "":
        return "not judged"
    return f"{'leaked' if leaked in (True, 'True') else 'no leak'}, accurate: {accurate}"


def markdown(table: list[dict]) -> str:
    lines = ["# Preset selection: plain model vs Dateline", "",
             f"Model {config.ANSWER_MODEL}; judges {config.JUDGE_MODEL}. keep = plain answer leaked or wrong, "
             "and Dateline's answer clean, accurate and at least 2 sentences. near miss = plain answer bad, "
             "Dateline partly accurate.", "",
             "| Row | Story | Cutoff | Question | Plain model | Plain verdict | Dateline | Dateline verdict "
             "| Sentences / groups | Keep |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in table:
        mark = "**keep**" if r["keep"] in (True, "True") else ("near miss" if r["near_miss"] in (True, "True") else "")
        lines.append(" | ".join([
            f"| {r['row_id']}", r["event"], r["cutoff"], _cell(r["question"]), _cell(r["plain_answer"], EXCERPT),
            _verdict(r["plain_leaked"], r["plain_accurate"]),
            _cell(r["pipeline_answer"] or r.get("notice", ""), EXCERPT),
            _verdict(r["pipeline_leaked"], r["pipeline_accurate"]), f"{r['n_sentences']} / {r['n_groups']}",
            f"{mark} |"]))
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", default="", help="comma-separated row ids (P01...)")
    args = parser.parse_args()
    todo = rows()
    if args.only:
        wanted = set(args.only.split(","))
        todo = [r for r in todo if r["row_id"] in wanted]
    print(f"{len(todo)} candidates, model {config.ANSWER_MODEL}", flush=True)
    table = []
    for n, row in enumerate(todo, 1):
        out = score(row)
        table.append(out)
        print(f"[{n}/{len(todo)}] {row['row_id']} {row['event']} {row['cutoff']} {row['question']!r} | plain "
              f"{_verdict(out['plain_leaked'], out['plain_accurate'])} | dateline "
              f"{_verdict(out['pipeline_leaked'], out['pipeline_accurate'])}, {out['n_sentences']} sentences"
              f"{' | KEEP' if out['keep'] is True else ''}", flush=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (config.RESULTS_DIR / "preset_selection.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(table)
    (config.RESULTS_DIR / "preset_selection.md").write_text(markdown(table), encoding="utf-8")
    kept = {}
    for r in table:
        kept[r["event"]] = kept.get(r["event"], 0) + (r["keep"] is True)
    print(f"\nwrote {config.RESULTS_DIR / 'preset_selection.csv'} and .md; kept per story: {kept}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
