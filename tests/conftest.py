"""Every test runs offline and on the stand-in engine, whatever the machine's .env says.

dateline/config.py reads .env at import, so on a machine set up for the demo (DATELINE_ENGINE=real)
unstubbed tests would otherwise call the real engine and the model. Tests that need another engine or a
model reply stub them themselves (server.get_answer, llm.complete, llm._call, litellm.completion).
"""
from __future__ import annotations

import sys
import types

import pytest

from dateline import config


def _no_network(*_args, **_kwargs):
    raise AssertionError("network call in tests")


@pytest.fixture(autouse=True)
def offline_fake_engine(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ENGINE", "fake")
    monkeypatch.setattr(config, "ENGINE_MODULE", "pipeline.engine")
    # The model cache, answer cache and usage log live beside CACHE_DIR: keep them out of the repo's cache/.
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache" / "llm")
    # The one place a model is really called is litellm.completion (dateline/llm.py imports it lazily).
    litellm = sys.modules.get("litellm")
    if litellm is not None:
        monkeypatch.setattr(litellm, "completion", _no_network)
    else:
        monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=_no_network))
