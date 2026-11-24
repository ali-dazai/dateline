import json
from datetime import date

import pandas as pd
import pytest

from dateline import llm
from eval import baseline, judges
from eval.report import summarise, wilson
from eval.testset import COLUMNS, check, cutoff_phrase


def test_parse_json_handles_fences_and_prose():
    assert llm.parse_json('```json\n{"leaked": true}\n```') == {"leaked": True}
    assert llm.parse_json('Sure: {"a": 1} done') == {"a": 1}


def test_parse_json_takes_the_first_of_two_objects():
    """A judge reply seen on the real model: an object, a blank line, then a second object."""
    assert llm.parse_json('{"leaked": true, "evidence": "x"}\n\n{"leaked": false}') == {"leaked": True, "evidence": "x"}
    assert llm.parse_json('Answer: {"a": {"b": 1}} and {"c": 2}') == {"a": {"b": 1}}
    with pytest.raises(ValueError):
        llm.parse_json("no json here")


def test_extract_quotes():
    text = 'The Sun said "ALL SAVED FROM TITANIC AFTER COLLISION" and "short".'
    assert baseline.extract_quotes(text) == ["ALL SAVED FROM TITANIC AFTER COLLISION"]


def test_cutoff_phrase():
    assert cutoff_phrase(date(1912, 4, 15)) == "the end of Monday, April 15, 1912"


def test_judges_parse_model_output(monkeypatch):
    monkeypatch.setattr(llm, "complete", lambda *a, **k: '{"leaked": true, "facts_found": ["1,500 died"], "evidence": "1,500"}')
    out = judges.leakage_judge("q", date(1912, 4, 15), "about 1,500 died", ["1,500 died"])
    assert out["leaked"] is True and out["facts_found"] == ["1,500 died"]
    monkeypatch.setattr(llm, "complete", lambda *a, **k: '{"accurate": "MAYBE", "refused": false}')
    assert judges.accuracy_judge("q", date(1912, 4, 15), "x", "y")["accurate"] == "no"


def test_leakage_prompt_includes_period_answer(monkeypatch):
    seen = {}

    def fake(prompt, **k):
        seen["prompt"] = prompt
        return '{"leaked": false}'

    monkeypatch.setattr(llm, "complete", fake)
    judges.leakage_judge("q", date(1912, 4, 16), "about 1,300 lost", ["1,500 died"], "Papers put the dead at 1,300.")
    assert "Papers put the dead at 1,300." in seen["prompt"] and "NOT hindsight" in seen["prompt"]
    judges.leakage_judge("q", date(1912, 4, 16), "x", ["y"])
    assert "What newspapers had reported" not in seen["prompt"]


def test_judges_can_be_told_a_minute(monkeypatch):
    """The minute story's judges see the minute, not "the end of" its day."""
    prompts = []
    monkeypatch.setattr(llm, "complete", lambda prompt, **k: prompts.append(prompt) or '{"leaked": false, "accurate": "yes"}')
    when = "8:00 a.m. EDT, Wednesday, June 21, 2023"
    judges.leakage_judge("q", date(2023, 6, 21), "a", ["b"], "c", when=when)
    judges.accuracy_judge("q", date(2023, 6, 21), "a", "c", when=when)
    assert all(when in p and "the end of" not in p for p in prompts)
    judges.accuracy_judge("q", date(1912, 4, 15), "a", "c")
    assert "the end of Monday, April 15, 1912" in prompts[-1]


def test_answer_at_tells_the_model_the_moment(monkeypatch):
    seen = []

    def fake(question, cutoff, model=None, *, when=None, **limits):
        seen.append((question, cutoff, when, limits))
        return "reply"

    monkeypatch.setattr(baseline, "answer", fake)
    assert baseline.answer_at("q", "titanic-1912", "1912-04-15") == "reply"
    assert baseline.answer_at("q", "titan-2023", "2023-06-21T14:00:00-04:00", attempts=2) == "reply"
    assert seen[0] == ("About the Titanic (April 1912): q", date(1912, 4, 15), None, {})
    assert seen[1][0] == "About the Titan submersible (June 2023): q"
    assert str(seen[1][1]) == "2023-06-21 18:00:00+00:00"
    assert seen[1][2] == "2:00 p.m. EDT, Wednesday, June 21, 2023" and seen[1][3] == {"attempts": 2}


def test_empty_answer_is_not_a_leak():
    assert judges.leakage_judge("q", date(1912, 4, 15), "  ", ["x"])["leaked"] is False


def test_testset_check_catches_problems():
    df = pd.DataFrame([dict.fromkeys(COLUMNS, "")])
    df.loc[0, ["question_id", "question", "cutoff"]] = ["Q1", "q", "15/04/1912"]
    problems = check(df)
    assert any("YYYY-MM-DD" in p for p in problems)
    assert any("future_facts" in p for p in problems)


def test_wilson():
    lo, hi = wilson(5, 10)
    assert 0 < lo < 0.5 < hi < 1


def test_summary_counts_leakage_only_on_memorised_questions():
    results = pd.DataFrame({
        "question_id": ["Q1", "Q2"], "cutoff": ["1912-04-15"] * 2, "setup": ["prompt_only"] * 2,
        "leaked": ["True", "False"], "accurate": ["yes", "no"], "refused": ["False", "False"],
    })
    tests = pd.DataFrame({"question_id": ["Q1", "Q2"], "cutoff": ["1912-04-15"] * 2, "memorised": ["yes", "no"]})
    s = summarise(results, tests)
    assert s.loc[0, "leakage_n"] == 1 and s.loc[0, "leakage_rate"] == 1.0
    assert s.loc[0, "accuracy_yes"] == pytest.approx(0.5)


def test_cache_avoids_second_call(tmp_path, monkeypatch):
    from dateline import config
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(config, "MIN_INTERVAL_S", 0)
    calls = []
    monkeypatch.setattr(llm, "_call", lambda *a, **k: calls.append(1) or "hi")
    assert llm.complete("x", model="m") == "hi"
    assert llm.complete("x", model="m") == "hi"
    assert len(calls) == 1


def test_the_plain_comparison_is_told_the_story(monkeypatch):
    """The web app's plain answer names the story, as Dateline's search does; the evaluation's does not."""
    prompts = []
    monkeypatch.setattr(baseline.llm, "complete", lambda prompt, **k: prompts.append(prompt) or "x")
    baseline.answer_at("Will they be found in time?", "titan-2023", "2023-06-21T14:00:00-04:00")
    baseline.answer_at("Who will buy it?", "lehman-2008", "2008-09-12")
    baseline.answer("Will they be found in time?", date(1912, 4, 15))
    assert prompts == ["About the Titan submersible (June 2023): Will they be found in time?",
                       "About Lehman Brothers (September 2008): Who will buy it?",
                       "Will they be found in time?"]


def _runs(setup, rows):
    """rows: (question_id, answer, leaked, accurate, n_removed)."""
    return pd.DataFrame([{"question_id": q, "cutoff": "1912-04-15", "setup": setup, "answer": a,
                          "leaked": l, "accurate": acc, "refused": "False", "n_removed": rm}
                         for q, a, l, acc, rm in rows])


def test_summary_reports_empty_answers_and_leakage_over_non_empty():
    from eval.report import summarise
    results = _runs("rag_guard", [
        ("Q01", "It sank.", "True", "no", "0"),
        ("Q02", "All safe.", "False", "yes", "1"),
        ("Q03", "", "False", "no", "2"),       # every sentence removed
        ("Q04", "", "False", "no", ""),        # nothing composed
    ])
    tests = pd.DataFrame({"question_id": ["Q01", "Q02", "Q03", "Q04"], "cutoff": ["1912-04-15"] * 4,
                          "memorised": ["yes"] * 4})
    s = summarise(results, tests).iloc[0]
    assert s["leakage_n"] == 4 and s["leakage_rate"] == pytest.approx(0.25)
    assert s["leakage_nonempty_n"] == 2 and s["leakage_nonempty_rate"] == pytest.approx(0.5)
    assert s["empty_or_removed_n"] == 2 and s["empty_or_removed"] == pytest.approx(0.5)
    assert s["all_removed_n"] == 1
    assert (s["accuracy_yes_n"], s["accuracy_partial_n"], s["accuracy_no_n"]) == (1, 0, 3)
    assert 0 <= s["accuracy_ci_low"] < 0.25 < s["accuracy_ci_high"] <= 1


def test_split_summary_separates_tuning_and_held_out_rows():
    from eval.report import markdown, split_summary, summarise
    results = pd.concat([
        _runs("prompt_only", [("Q01", "x", "True", "yes", ""), ("Q08", "x", "True", "yes", "")]),
        _runs("rag", [("Q01", "x", "True", "yes", "0"), ("Q07", "x", "False", "no", "0"),
                      ("Q08", "x", "False", "yes", "0")]),
        _runs("rag_guard_no_expansion", [("Q09", "", "False", "no", "1")]),
    ])
    tests = pd.DataFrame({"question_id": ["Q01", "Q07", "Q08", "Q09"], "cutoff": ["1912-04-15"] * 4,
                          "memorised": ["yes"] * 4})
    split = split_summary(results, tests)
    assert "Prompt only" not in set(split["setup"])
    rag = split[split["setup"] == "RAG"].set_index("rows")
    assert rag.loc["tuning (Q01-Q07)", "runs"] == 2 and rag.loc["tuning (Q01-Q07)", "leaks"] == 1
    assert rag.loc["held-out (Q08-Q15)", "runs"] == 1 and rag.loc["held-out (Q08-Q15)", "leaks"] == 0
    ablation = split[split["setup"] == "RAG + quote check, no term expansion"]
    assert list(ablation["rows"]) == ["held-out (Q08-Q15)"]
    text = markdown(summarise(results, tests), split)
    assert "Tuning rows and held-out rows" in text and "1/2 (50%)" in text


def test_ablation_settings_are_restored(monkeypatch):
    import os
    from eval import run_experiment
    monkeypatch.setenv("DATELINE_RERANK", "1")
    monkeypatch.delenv("DATELINE_TERM_EXPANSION", raising=False)
    with run_experiment._settings({"DATELINE_RERANK": "0", "DATELINE_TERM_EXPANSION": "0"}):
        assert os.environ["DATELINE_RERANK"] == "0" and os.environ["DATELINE_TERM_EXPANSION"] == "0"
    assert os.environ["DATELINE_RERANK"] == "1" and "DATELINE_TERM_EXPANSION" not in os.environ


def test_ablation_setups_run_guarded_with_their_switch(monkeypatch):
    import os
    from eval import run_experiment
    seen = {}

    def fake_answer(question, cutoff, guard=True, event="titanic-1912"):
        seen.update(guard=guard, expansion=os.environ.get("DATELINE_TERM_EXPANSION"),
                    rerank=os.environ.get("DATELINE_RERANK"))
        return {"sentences": [], "removed": [], "groups": [], "retrieved": []}

    monkeypatch.setattr(run_experiment, "get_answer", lambda: fake_answer)
    monkeypatch.setattr(run_experiment, "leakage_judge", lambda *a, **k: {"leaked": False, "facts_found": [],
                                                                           "evidence": ""})
    monkeypatch.setattr(run_experiment, "accuracy_judge", lambda *a, **k: {"accurate": "no", "refused": True,
                                                                            "reason": ""})
    monkeypatch.delenv("DATELINE_TERM_EXPANSION", raising=False)
    row = pd.Series({"question_id": "Q01", "cutoff": "1912-04-15", "question": "Is the Titanic safe?",
                     "future_facts": "", "period_answer": ""})
    record = run_experiment.run_one(row, "rag_guard_no_expansion")
    assert seen == {"guard": True, "expansion": "0", "rerank": "1"}
    assert record["setup"] == "rag_guard_no_expansion" and "DATELINE_TERM_EXPANSION" not in os.environ
    assert json.loads(record["steps"]) == {"DATELINE_RERANK": "1", "DATELINE_TERM_EXPANSION": "0"}


def test_main_setups_force_every_step_on(monkeypatch):
    import os
    from eval import run_experiment
    seen = []

    def fake_answer(question, cutoff, guard=True, event="titanic-1912"):
        seen.append((os.environ.get("DATELINE_TERM_EXPANSION"), os.environ.get("DATELINE_RERANK")))
        return {"sentences": [], "removed": [], "groups": [], "retrieved": []}

    monkeypatch.setattr(run_experiment, "get_answer", lambda: fake_answer)
    monkeypatch.setattr(run_experiment, "leakage_judge", lambda *a, **k: {"leaked": False, "facts_found": [],
                                                                           "evidence": ""})
    monkeypatch.setattr(run_experiment, "accuracy_judge", lambda *a, **k: {"accurate": "no", "refused": True,
                                                                            "reason": ""})
    monkeypatch.setenv("DATELINE_TERM_EXPANSION", "0")  # a stray setting left in the shell or .env
    monkeypatch.setenv("DATELINE_RERANK", "off")
    row = pd.Series({"question_id": "Q01", "cutoff": "1912-04-15", "question": "Is the Titanic safe?",
                     "future_facts": "", "period_answer": ""})
    for setup in ("rag", "rag_guard"):
        record = run_experiment.run_one(row, setup)
        assert json.loads(record["steps"]) == {"DATELINE_RERANK": "1", "DATELINE_TERM_EXPANSION": "1"}
    assert seen == [("1", "1"), ("1", "1")]
    assert os.environ["DATELINE_TERM_EXPANSION"] == "0" and os.environ["DATELINE_RERANK"] == "off"


def test_old_results_file_gains_the_new_column(tmp_path, monkeypatch):
    from eval import run_experiment
    old = [c for c in run_experiment.RESULT_COLUMNS if c != "steps"]
    out = tmp_path / "results.csv"
    pd.DataFrame([{c: "x" for c in old} | {"question_id": "Q01", "cutoff": "1912-04-15", "setup": "rag"}]).to_csv(
        out, index=False)
    tests = pd.DataFrame([{"question_id": "Q01", "cutoff": "1912-04-15", "question": "q", "future_facts": "",
                           "period_answer": ""}])
    monkeypatch.setattr(run_experiment, "load", lambda path: tests)
    monkeypatch.setattr(run_experiment, "get_answer", lambda: lambda q, c, g=True, event="titanic-1912": {
        "sentences": [], "removed": [], "groups": [], "retrieved": []})
    monkeypatch.setattr(run_experiment, "leakage_judge", lambda *a, **k: {"leaked": False, "facts_found": [],
                                                                           "evidence": ""})
    monkeypatch.setattr(run_experiment, "accuracy_judge", lambda *a, **k: {"accurate": "no", "refused": True,
                                                                            "reason": ""})
    monkeypatch.setattr("sys.argv", ["run_experiment", "--out", str(out), "--setups", "rag"])
    run_experiment.main()
    assert list(pd.read_csv(out, dtype=str).columns) == run_experiment.RESULT_COLUMNS
    assert len(pd.read_csv(out, dtype=str)) == 1  # the row was already done
