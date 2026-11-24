from datetime import date

from dateline.contract import answer_text, validate_result
from dateline.fake_engine import answer


def test_fake_engine_meets_contract_with_guard():
    for d in (date(1912, 4, 14), date(1912, 4, 15), date(1912, 4, 18)):
        result = answer("What happened to the Titanic?", d, True)
        assert validate_result(result, d) == []


def test_guard_off_lets_hindsight_through_and_guard_on_removes_it():
    d = date(1912, 4, 15)
    off = answer("What happened to the Titanic?", d, False)
    on = answer("What happened to the Titanic?", d, True)
    assert "1,500" in answer_text(off)
    assert "1,500" not in answer_text(on)
    assert on["removed"]


def test_validator_flags_articles_after_cutoff():
    result = answer("q", date(1912, 4, 18), True)
    assert any("after the cutoff" in p for p in validate_result(result, date(1912, 4, 15)))


def test_optional_fields_are_optional():
    sentence = {"text": "t", "quote": "a quote of six words here", "article_id": "a1", "paper": "p",
                "date": "1912-04-15", "page": "1", "url": "u"}
    group = {"claim": "c", "count": 1, "article_ids": ["a1"]}
    plain = {"sentences": [sentence], "removed": [], "groups": [group], "retrieved": ["a1"]}
    assert validate_result(plain, date(1912, 4, 15)) == []
    full = {"sentences": [{**sentence, "flag": "why", "group": 0, "published": "1912-04-15T12:00:00Z"}],
            "removed": [], "groups": [{**group, "papers": ["p"], "estimated": True}], "retrieved": ["a1"],
            "notice": "n"}
    assert validate_result(full, date(1912, 4, 15)) == []
