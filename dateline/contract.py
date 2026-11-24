"""The engine interface: the one function the web app and the evaluation call.

answer(question: str, cutoff: date | datetime, guard: bool = True,
       event: str = "titanic-1912") -> AnswerResult

The front end and the experiment only ever call this function, via engine.get_answer().
validate_result() checks any engine's output against the contract.

Version 2 (proposed 2026-11-02; see docs/engine-interface.md) adds, for the three demo stories:
  * event      which story's corpus to search (see dateline/events.py)
  * cutoff     a datetime for the minute-level 2023 story; a date still means end of day
  * Sentence.published  ISO time the source counts from (available_at for Guardian items)
  * Sentence.group      index into groups (-1 if the sentence's source is in no group)
  * Sentence.flag       why the quote check would remove the sentence (kept because guard=False)
  * Group.papers        the distinct papers behind a claim (count is their number)
  * Group.estimated     the count rests partly on a model's reading of article openings
  * AnswerResult.notice a message the reader must see: nothing available yet, nothing found,
                        the model failed, or every sentence removed
Per-day / per-hour article counts come from events.counts(), served at GET /api/range.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Callable, NotRequired, TypedDict


class Sentence(TypedDict):
    text: str        # the sentence shown to the user
    quote: str       # verbatim quote from the article that backs it
    article_id: str
    paper: str
    date: str        # YYYY-MM-DD, must be <= cutoff
    page: str
    url: str         # 1912: https://www.loc.gov/item/{lccn}/{date}/ed-{edition}/ ; Guardian: article URL
    published: NotRequired[str]  # ISO 8601 UTC; must be <= cutoff (minute story)
    group: NotRequired[int]      # index into AnswerResult.groups, -1 if none
    flag: NotRequired[str]       # guard=False only: why the quote check would have removed it


class Removed(TypedDict):
    text: str
    reason: str      # why the quote check deleted it


class Group(TypedDict):
    claim: str
    count: int
    article_ids: list[str]
    papers: NotRequired[list[str]]  # distinct papers behind the claim
    estimated: NotRequired[bool]    # True when the count rests partly on classified openings


class AnswerResult(TypedDict):
    sentences: list[Sentence]
    removed: list[Removed]
    groups: list[Group]
    retrieved: list[str]
    notice: NotRequired[str]  # shown to the reader, e.g. "No articles had been published by this moment."


AnswerFn = Callable[..., AnswerResult]

# Required keys only. Listed by hand: with postponed annotations, TypedDict cannot see NotRequired.
_SENTENCE_KEYS = {k for k in Sentence.__annotations__ if k not in ("published", "group", "flag")}
_REMOVED_KEYS = set(Removed.__annotations__)
_GROUP_KEYS = {k for k in Group.__annotations__ if k not in ("papers", "estimated")}


def _as_datetime(value) -> datetime:
    from .events import parse_cutoff

    return parse_cutoff("", value)


def validate_result(result: dict, cutoff: date | datetime) -> list[str]:
    """Return a list of contract violations (empty list = valid)."""
    limit = _as_datetime(cutoff)
    problems: list[str] = []
    for key in ("sentences", "removed", "groups", "retrieved"):
        if key not in result:
            problems.append(f"missing key: {key}")
    n_groups = len(result.get("groups", []))
    for i, s in enumerate(result.get("sentences", [])):
        missing = _SENTENCE_KEYS - set(s)
        if missing:
            problems.append(f"sentences[{i}] missing {sorted(missing)}")
            continue
        stamp = s.get("published") or s["date"]
        try:
            if _as_datetime(stamp) > limit:
                problems.append(f"sentences[{i}] cites an article dated after the cutoff ({stamp})")
        except ValueError:
            problems.append(f"sentences[{i}] has a bad date: {stamp!r}")
        if not s["quote"].strip():
            problems.append(f"sentences[{i}] has an empty quote")
        if "group" in s and not (s["group"] == -1 or 0 <= s["group"] < n_groups):
            problems.append(f"sentences[{i}] group {s['group']} is not an index into groups")
    for i, r in enumerate(result.get("removed", [])):
        if _REMOVED_KEYS - set(r):
            problems.append(f"removed[{i}] missing {sorted(_REMOVED_KEYS - set(r))}")
    for i, g in enumerate(result.get("groups", [])):
        if _GROUP_KEYS - set(g):
            problems.append(f"groups[{i}] missing {sorted(_GROUP_KEYS - set(g))}")
    return problems


def answer_text(result: AnswerResult) -> str:
    """The answer as plain text (used by the judges)."""
    return " ".join(s["text"].strip() for s in result["sentences"]).strip()
