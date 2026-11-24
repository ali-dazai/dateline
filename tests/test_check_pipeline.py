"""The real-data check's tuning set stays the one docs/pipeline-design.md reports ("Real-data check and
tuning")."""
import csv
import importlib.util
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("check_pipeline", ROOT / "scripts" / "check_pipeline.py")
check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check)


def _committed_candidates(rev: str) -> list[dict]:
    text = subprocess.run(["git", "show", f"{rev}:data/preset_candidates.csv"], cwd=ROOT, capture_output=True,
                          text=True, encoding="utf-8", check=True).stdout
    return list(csv.DictReader(text.splitlines()))


def test_tuning_set_is_the_first_30_candidates_and_15_test_rows():
    rows = check._rows("tuning")
    presets = [r for r in rows if r["row_id"].startswith("P")]
    assert len(rows) == 45 and len(presets) == 30
    assert [r["row_id"] for r in presets] == [f"P{i:02d}" for i in range(1, 31)]
    assert all("@" in r["row_id"] for r in rows[30:])


def test_tuning_candidates_match_those_the_tuning_ran_on():
    try:
        before = _committed_candidates("54a3aeb")  # the candidates file Task 10's tuning read
    except (subprocess.CalledProcessError, FileNotFoundError):
        import pytest
        pytest.skip("git history not available")
    presets = [r for r in check._rows("tuning") if r["row_id"].startswith("P")]
    assert [(r["event"], r["cutoff"], r["question"]) for r in presets] == \
        [(r["event"], r["cutoff"], r["question"]) for r in before]
