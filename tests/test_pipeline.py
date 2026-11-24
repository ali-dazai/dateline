"""End-to-end run of the experiment and report on the fake engine, with the LLM mocked."""
import sys

import pandas as pd

from dateline import config, llm
from eval import report, run_experiment
from eval.testset import COLUMNS


def fake_complete(prompt, *, json_mode=False, **kwargs):
    if json_mode and "hindsight" in prompt:
        return '{"leaked": %s, "facts_found": [], "evidence": ""}' % ("true" if "1,500" in prompt.split("<<<")[-1] else "false")
    if json_mode:
        return '{"accurate": "yes", "refused": false, "reason": "matches"}'
    return 'The papers say "ALL SAVED FROM TITANIC AFTER COLLISION".'


def test_full_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "complete", fake_complete)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    row = dict.fromkeys(COLUMNS, "")
    row.update(question_id="Q01", question="What happened to the Titanic?", cutoff="1912-04-15",
               period_answer="Papers report all passengers saved.", future_facts="about 1,500 died",
               memorised="yes", verified_by="tester")
    pd.DataFrame([row]).to_csv(tmp_path / "testset.csv", index=False)
    monkeypatch.setattr(sys, "argv", ["run", "--testset", str(tmp_path / "testset.csv"),
                                      "--out", str(tmp_path / "results.csv")])
    run_experiment.main()
    res = pd.read_csv(tmp_path / "results.csv")
    assert set(res["setup"]) == {"prompt_only", "rag", "rag_guard"}
    leaked = dict(zip(res["setup"], res["leaked"]))
    assert leaked == {"prompt_only": False, "rag": True, "rag_guard": False}
    run_experiment.main()  # resumable: nothing new to run
    assert len(pd.read_csv(tmp_path / "results.csv")) == 3
    report.main()
    assert (tmp_path / "chart.png").exists()
