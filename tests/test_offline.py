"""The test suite never reaches a model or the engine named in .env (tests/conftest.py)."""
import pytest

from dateline import config, llm
from dateline.engine import get_answer
from dateline.fake_engine import answer as fake_answer


def test_a_real_model_call_fails_loudly():
    with pytest.raises(AssertionError, match="network call in tests"):
        llm.complete("hello", use_cache=False, attempts=1)


def test_tests_use_the_stand_in_engine():
    assert config.ENGINE == "fake" and get_answer() is fake_answer


def test_preload_imports_the_model_library(monkeypatch):
    import sys
    import types

    fake = types.ModuleType("litellm")
    monkeypatch.setitem(sys.modules, "litellm", fake)
    assert llm.preload() is fake
