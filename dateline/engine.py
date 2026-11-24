"""Selects the engine the app and the experiment call: the fake one or the real one (DATELINE_ENGINE)."""
from __future__ import annotations

import importlib
import inspect

from . import config
from .contract import AnswerFn
from .events import DEFAULT_EVENT


def _with_event(fn) -> AnswerFn:
    """Let an engine written before v2 (no `event` argument) keep working for the 1912 story."""
    if "event" in inspect.signature(fn).parameters:
        return fn

    def call(question, cutoff, guard=True, event=DEFAULT_EVENT):
        if event != DEFAULT_EVENT:
            raise NotImplementedError(f"this engine only supports {DEFAULT_EVENT}")
        return fn(question, cutoff, guard)

    return call


def get_answer() -> AnswerFn:
    """Return answer(question, cutoff, guard, event) for the engine set by DATELINE_ENGINE."""
    if config.ENGINE == "fake":
        from .fake_engine import answer

        return answer
    module = importlib.import_module(config.ENGINE_MODULE)
    return _with_event(module.answer)
