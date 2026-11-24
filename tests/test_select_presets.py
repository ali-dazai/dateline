"""The preset keep rule and the selection table (scripts/select_presets.py), with no model."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("select_presets", ROOT / "scripts" / "select_presets.py")
select = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(select)


@pytest.mark.parametrize("plain, pipe, n, keep, near", [
    ({"leaked": True, "accurate": "yes"}, {"leaked": False, "accurate": "yes"}, 2, True, False),
    ({"leaked": False, "accurate": "no"}, {"leaked": False, "accurate": "yes"}, 3, True, False),
    ({"leaked": False, "accurate": "partial"}, {"leaked": False, "accurate": "yes"}, 3, False, False),
    ({"leaked": True, "accurate": "no"}, {"leaked": False, "accurate": "yes"}, 1, False, False),
    ({"leaked": True, "accurate": "no"}, {"leaked": True, "accurate": "yes"}, 4, False, False),
    ({"leaked": True, "accurate": "no"}, {"leaked": False, "accurate": "partial"}, 4, False, True),
])
def test_keep_rule(plain, pipe, n, keep, near):
    out = select.decide(plain, pipe, n)
    assert out["keep"] is keep and out["near_miss"] is near
    assert out["plain_bad"] is (plain["leaked"] or plain["accurate"] == "no")


def test_markdown_table_escapes_and_marks_rows():
    rows = [{"row_id": "P01", "event": "titanic-1912", "cutoff": "1912-04-15", "question": "Safe?",
             "plain_answer": "All | saved\nyes", "plain_leaked": True, "plain_accurate": "no",
             "pipeline_answer": "The Sun said all were saved.", "pipeline_leaked": False,
             "pipeline_accurate": "yes", "n_sentences": 2, "n_groups": 1, "keep": True, "near_miss": False}]
    text = select.markdown(rows)
    assert "| P01 |" in text and "All \\| saved yes" in text and "**keep**" in text
    assert "leaked, accurate: no" in text
