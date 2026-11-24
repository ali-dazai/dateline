"""Setup 1: prompt only. The model is TOLD the date and asked to answer as of then.

No retrieval, no quote check. This is the baseline the research says will leak hindsight.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

from dateline import config, events, llm
from eval.testset import cutoff_phrase

SYSTEM = (
    "You are a newspaper reader living in {when}. You have read every newspaper printed "
    "up to and including that day, and you know nothing that happened or was reported later. "
    "Answer the question using only what was known at that time. If newspapers disagreed, say so. "
    "When you rely on a specific newspaper, quote its words in double quotes and name the paper."
)


def answer(question: str, cutoff: date, model: str | None = None, *, when: str | None = None,
           **limits: Any) -> str:
    """The model's answer when told it is `cutoff` (a day: "the end of Monday, April 15, 1912").

    `when` replaces the day phrase with other moment text (the minute story passes
    pipeline.engine.moment()). `limits` (attempts, timeout_s, deadline_s) go to llm.complete; the
    evaluation passes none and keeps the batch retries."""
    when = cutoff_phrase(cutoff) if when is None else when
    return llm.complete(
        question,
        system=SYSTEM.format(when=when),
        model=model or config.ANSWER_MODEL,
        **limits,
    ).strip()


def about(event: str, question: str) -> str:
    """The question with its story named, neutrally: "About the Titanic (April 1912): <question>".

    The page's questions often leave the story out ("Will they be found in time?"); Dateline's search adds
    the story's terms, so the plain comparison is told the story too. Only the month the story starts is
    given, nothing about how it ends."""
    spec = events.get(event)
    start = events.parse_cutoff(event, spec["start"])
    return f"About {spec.get('subject') or spec['title']} ({start:%B %Y}): {question}"


def answer_at(question: str, event: str, cutoff: str | date | datetime, **limits: Any) -> str:
    """The plain answer the web app shows (/api/plain) for a story's cutoff, the question naming the story.

    The system prompt is the evaluation's: day stories are told the end of that day; the minute story is
    told the minute (pipeline.engine.moment(), in the story's display time). The evaluation itself calls
    answer() with the bare question, so its prompts are unchanged."""
    from pipeline.engine import moment  # noqa: PLC0415 - pipeline.engine imports eval.testset

    at = events.parse_cutoff(event, cutoff)
    asked = about(event, question)
    if events.get(event)["granularity"] == "day":
        return answer(asked, at.date(), **limits)
    return answer(asked, at, when=moment(event, at), **limits)


_QUOTE = re.compile(r"[\"“]([^\"”]{12,300})[\"”]")


def extract_quotes(text: str) -> list[str]:
    """Quoted passages (12+ characters) the model claims came from newspapers."""
    return [q.strip() for q in _QUOTE.findall(text)]
