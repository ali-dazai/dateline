"""The engine: `answer()` runs search, re-rank, compose and the checks, and returns the contract result.

One question costs up to four model calls (search terms, re-rank, compose, classify). Everything the
model produced is cached on disk before the quote check runs, so the check on and off share one model
answer and a repeated question is served without search or model calls. Cases the reader must see
(nothing available yet, nothing to search, nothing found, the model failed, every sentence removed) come
back as a result with a `notice`, never as an exception.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import sqlite3
import tempfile
import threading
import time
from collections.abc import Iterator, Mapping
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from dateline import config, events
from dateline.contract import AnswerResult
from dateline.events import DEFAULT_EVENT
from eval.testset import cutoff_phrase
from pipeline import compose as _compose
from pipeline import groups as _groups
from pipeline import index as _index
from pipeline import passages as _passages
from pipeline import query as _query
from pipeline import rerank as _rerank
from pipeline import search as _search
from pipeline import verify as _verify

SEARCH_K = 50
KEEP_TOP = 3
INTERACTIVE_BUDGET_S = 30.0
INTERACTIVE_CALL_S = 20.0
INTERACTIVE_ATTEMPTS = 2
CLASSIFY_CUTOFF_S = 20.0

NOTICE_NO_INDEX = "The search index for this story has not been built yet."
NOTICE_NO_ARTICLES = "No articles had been published by this moment."
NOTICE_NO_WORDS = "The question has no words to search for."
NOTICE_NOTHING_FOUND = "Nothing published by this moment answers this question."
NOTICE_MODEL_FAILED = "The model could not produce an answer. Try again."
NOTICE_ALL_REMOVED = "Every sentence was removed by the quote check."

_WORDLIKE = re.compile(r"[^\W_]")

# One INFO record per step of an answer ("terms", "search", "rerank", "passages", "compose", "classify",
# "checks"), with the step's name and duration in the record's `step` and `seconds` attributes, so a
# caller can attach a handler and see where the time goes.
log = logging.getLogger("pipeline")
_PUNCTUATION = re.compile(r"[^\w\s]")


def _pipeline_version() -> str:
    """sha256 over every pipeline/*.py file (name and bytes, sorted by name)."""
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py"), key=lambda p: p.name):
        digest.update(path.name.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


PIPELINE_VERSION = _pipeline_version()


def _now() -> float:
    return time.monotonic()


@contextlib.contextmanager
def _timed(step: str) -> Iterator[None]:
    """Log how long the block took as one INFO record on the "pipeline" logger."""
    started = time.perf_counter()
    try:
        yield
    finally:
        seconds = time.perf_counter() - started
        log.info("step %s took %.3f s", step, seconds, extra={"step": step, "seconds": seconds})


# --- the moment in words -----------------------------------------------------------------------------

def _display_zone(spec: dict) -> tuple[float, str]:
    offset = spec.get("display_offset_hours")
    if offset is None:
        return 0, "UTC"
    return offset, spec.get("display_tz_label") or f"UTC{offset:+g}"


def _clock(local: datetime) -> str:
    return f"{local.hour % 12 or 12}:{local.minute:02d} {'a.m.' if local.hour < 12 else 'p.m.'}"


def moment(event: str, cutoff: date | datetime) -> str:
    """The cutoff as the model and the reader are told it.

    Day stories: "the end of Monday, April 15, 1912" (the prompt-only baseline's wording). Minute stories:
    "10:00 a.m. EDT, Wednesday, June 21, 2023" in the story's display zone (a fixed offset from the event
    entry, as Windows Python has no time-zone database), or in UTC when the entry names none."""
    spec = events.get(event)
    at = events.parse_cutoff(event, cutoff)
    if spec["granularity"] == "day":
        return cutoff_phrase(at.date())
    offset, label = _display_zone(spec)
    local = at + timedelta(hours=offset)
    return f"{_clock(local)} {label}, {local:%A}, {local:%B} {local.day}, {local.year}"


def _display_times(event: str, articles: dict[str, dict]) -> list[str]:
    """Display-time strings of the given items ("1:39", "01:39", "a.m.", "EDT"), for minute stories."""
    spec = events.get(event)
    if spec["granularity"] == "day":
        return []
    offset, label = _display_zone(spec)
    out = [label]
    for art in articles.values():
        try:
            local = events.parse_cutoff(event, str(art.get("available_at") or "")) + timedelta(hours=offset)
        except ValueError:
            continue
        hour = local.hour % 12 or 12
        out += [f"{hour}:{local.minute:02d}", f"{hour:02d}:{local.minute:02d}",
                "a.m." if local.hour < 12 else "p.m."]
    return list(dict.fromkeys(out))


# --- interactive limits ------------------------------------------------------------------------------

class _Limits(Mapping):
    """Model-call limits for an interactive answer, read afresh each time a call unpacks them.

    Passed as `llm_opts`, so every call (including compose's retry) gets the budget that is left:
    attempts=2, timeout_s=min(20, remaining), deadline_s=remaining, out of 30 s for the whole answer.
    Values are computed on every read, so no lookup can return an older, larger budget."""

    _KEYS = ("attempts", "timeout_s", "deadline_s")

    def __init__(self, started: float) -> None:
        self._started = started

    def elapsed(self) -> float:
        return _now() - self._started

    def remaining(self) -> float:
        return max(0.0, INTERACTIVE_BUDGET_S - self.elapsed())

    def __getitem__(self, key: str) -> Any:
        if key == "attempts":
            return INTERACTIVE_ATTEMPTS
        if key == "timeout_s":
            return min(INTERACTIVE_CALL_S, self.remaining())
        if key == "deadline_s":
            return self.remaining()
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._KEYS)

    def __len__(self) -> int:
        return len(self._KEYS)


def _out_of_time(limits: _Limits | None) -> bool:
    return limits is not None and limits.remaining() <= 0


# --- answer cache ------------------------------------------------------------------------------------

_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(key: str) -> threading.Lock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.Lock())


def normalise_question(question: str) -> str:
    """Lower-cased, punctuation removed, whitespace collapsed; stop words kept ("not" changes the answer)."""
    return " ".join(_PUNCTUATION.sub("", question.lower()).split())


def switches() -> dict[str, bool]:
    """The pipeline steps that can be turned off for an ablation, read from the environment per answer:
    search-term expansion (DATELINE_TERM_EXPANSION) and the re-rank (DATELINE_RERANK), both on by default."""
    return {"term_expansion": _query.expansion_on(), "rerank": _rerank.rerank_on()}


def _cache_key(event: str, question: str, rowid: int, when: str, steps: Mapping[str, bool]) -> str:
    stat = _index.index_path(event).stat()
    payload = {"event": event, "question": normalise_question(question), "rowid": rowid,
               "model": config.ANSWER_MODEL, "index_size": stat.st_size, "index_mtime_ns": stat.st_mtime_ns,
               "moment": when, "version": PIPELINE_VERSION, "steps": dict(steps)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _cache_path(key: str) -> Path:
    return Path(config.CACHE_DIR).parent / "answers" / f"{key}.json"


_STATE_KEYS = {"retrieved", "chosen", "passages", "articles", "draft", "classified"}


def _load(key: str) -> dict | None:
    try:
        state = json.loads(_cache_path(key).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return state if isinstance(state, dict) and _STATE_KEYS <= set(state) else None


def _store(key: str, state: dict) -> None:
    """Write atomically (temp file + os.replace); a failed write only costs the cache."""
    path = _cache_path(key)
    tmp = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f"{key[:16]}.", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(state, fh, ensure_ascii=False)
        os.replace(tmp, path)
        tmp = None
    except OSError:
        pass
    finally:
        if tmp is not None:
            Path(tmp).unlink(missing_ok=True)


# --- the steps ---------------------------------------------------------------------------------------

def _distinct_claims(claims: dict[str, str | None]) -> list[str]:
    out: dict[str, str] = {}
    for label in claims.values():
        if isinstance(label, str) and label.strip():
            out.setdefault(" ".join(label.split()).lower(), label)
    return list(out.values())


def _run(con: sqlite3.Connection, question: str, limit: datetime, when: str, event: str,
         limits: _Limits | None, steps: Mapping[str, bool]) -> tuple[dict, list[str]]:
    """Search, re-rank, compose and classify: the pre-check state (draft None if no answer was composed)
    and the reasons the run fell short of a full answer (a model call that raised, or classify skipped
    for time). A run with reasons is not cached, so a later request gets a full answer."""
    opts: Mapping = limits if limits is not None else {}
    degraded: list[str] = []
    state = {"retrieved": [], "chosen": [], "passages": {}, "articles": {}, "draft": None, "classified": {}}
    if _out_of_time(limits):
        return state, ["out of time"]
    with _timed("terms"):
        terms = _query.search_terms(question, when, events.get(event).get("story_terms", ()), llm_opts=opts,
                                    on_fallback=degraded.append, expand=steps["term_expansion"])
    with _timed("search"):
        hits = _search.search(con, terms, limit, k=SEARCH_K)
        keep = _search.search(con, _query.question_terms(question), limit, k=KEEP_TOP)
        known = {h.article_id for h in hits}
        hits += [h for h in keep if h.article_id not in known]
        ids = [h.article_id for h in hits]
        state["retrieved"] = ids
        if hits:
            full = _index.get_articles(con, ids)
            openings = {aid: _passages.opening(str(full[aid].get("text") or "")) for aid in ids if aid in full}
    if not hits:
        state["draft"] = {"sentences": [], "claims": {}, "figures": []}
        return state, degraded

    if _out_of_time(limits):
        return state, [*degraded, "out of time"]
    with _timed("rerank"):
        chosen = _rerank.rerank(question, when, hits, openings, [h.article_id for h in keep], llm_opts=opts,
                                on_fallback=degraded.append, use_model=steps["rerank"])
    with _timed("passages"):
        passages = {aid: _passages.passage(str(full[aid].get("text") or ""), list(terms))
                    for aid in chosen if aid in full}
    given = {aid: full[aid] for aid in passages}
    state.update(chosen=list(passages), passages=passages, articles=given)
    if _out_of_time(limits):
        return state, [*degraded, "out of time"]
    try:
        with _timed("compose"):
            draft = _compose.compose(question, when, passages, given, llm_opts=opts)
    except _compose.ComposeError as err:
        return state, [*degraded, str(err)]

    claims = _distinct_claims(draft["claims"])
    classified: dict[str, str] = {}
    others = {aid: text for aid, text in openings.items() if aid not in passages}
    if len(claims) >= 2 and others:
        if limits is not None and limits.elapsed() >= CLASSIFY_CUTOFF_S:
            degraded.append("classify skipped for time")
        else:
            with _timed("classify"):
                assigned = _groups.classify(question, claims, others, llm_opts=opts,
                                            on_fallback=degraded.append)
            classified = {aid: claims[i] for aid, i in assigned.items() if i is not None}
    # Classified articles keep their full text: build_groups checks a label against it for `estimated`.
    state.update(draft=draft, classified=classified,
                 articles={**given, **{aid: full[aid] for aid in classified}})
    return state, degraded


def _empty(notice: str | None = None, retrieved: list[str] | None = None) -> AnswerResult:
    result: dict = {"sentences": [], "removed": [], "groups": [], "retrieved": list(retrieved or [])}
    if notice:
        result["notice"] = notice
    return result  # type: ignore[return-value]


def _finish(state: dict, question: str, limit: datetime, when: str, event: str, guard: bool) -> AnswerResult:
    """Apply the checks to the cached state and build the contract result."""
    retrieved, draft = state["retrieved"], state["draft"]
    if draft is None:
        return _empty(NOTICE_MODEL_FAILED, retrieved)
    if not retrieved:
        return _empty(NOTICE_NOTHING_FOUND)
    passages, articles = state["passages"], state["articles"]
    given_articles = {aid: articles[aid] for aid in passages if aid in articles}
    # The story's own name ("Titan", "Lehman") is known at every moment, so a sentence may use it.
    allowed = _verify.allowed_terms(question, when, given_articles, events.bounds(event)[0], limit,
                                    extra=[*_display_times(event, given_articles),
                                           *events.get(event).get("story_terms", ())])
    groups, group_of = _groups.build_groups(draft["claims"], articles, passages, draft["sentences"],
                                            question, allowed, state["classified"])
    given = set(passages)
    sentences, removed = [], []
    short = f"quote shorter than {_verify.MIN_QUOTE_WORDS} words"
    for draft_sentence in draft["sentences"]:
        aid = draft_sentence.get("article_id", "")
        out, reason = _verify.check(draft_sentence, passages.get(aid, ""), articles.get(aid), limit, given,
                                    question, allowed)
        if not _WORDLIKE.search(out["quote"]):  # the contract forbids empty quotes, guard or not
            removed.append({"text": out["text"], "reason": short})
            continue
        if reason is not None:
            if guard:
                removed.append({"text": out["text"], "reason": reason})
                continue
            out["flag"] = reason
        if not out.get("published"):
            out.pop("published", None)
        out["group"] = group_of.get(out["article_id"], -1) if out["article_id"] else -1
        sentences.append(out)

    result: dict = {"sentences": sentences, "removed": removed, "groups": groups, "retrieved": list(retrieved)}
    chart = _groups.build_chart(draft["figures"], passages, articles, group_of, question)
    if chart is not None:
        result["chart"] = chart
    if not draft["sentences"]:
        result["notice"] = NOTICE_NOTHING_FOUND
    elif not sentences:
        result["notice"] = NOTICE_ALL_REMOVED
    return result  # type: ignore[return-value]


def _locate(con: sqlite3.Connection, question: str, limit: datetime, event: str,
            steps: Mapping[str, bool]) -> tuple[str, str, str | None]:
    """(cache key, moment in words, None) for an answer at `limit`, or ("", "", notice) when none is needed."""
    rowid = _index.max_rowid(con, limit)
    if rowid == 0:
        return "", "", NOTICE_NO_ARTICLES
    if not _search.tokens(question):
        return "", "", NOTICE_NO_WORDS
    when = moment(event, limit)
    return _cache_key(event, question, rowid, when, steps), when, None


def stored(question: str, cutoff: date | datetime, event: str = DEFAULT_EVENT) -> bool:
    """True when `answer()` would find this question's model answer in the cache (no search, no model call).

    False when the index is missing, no article is available yet or the question has no searchable words:
    `answer()` returns a notice for those instead of an answer."""
    limit = events.parse_cutoff(event, cutoff)
    try:
        con = _index.connect(event)
    except _index.IndexMissing:
        return False
    try:
        key, _when, notice = _locate(con, question if isinstance(question, str) else str(question or ""), limit,
                                     event, switches())
    finally:
        con.close()
    return notice is None and _load(key) is not None


def answer(question: str, cutoff: date | datetime, guard: bool = True, event: str = DEFAULT_EVENT, *,
           interactive: bool = False) -> AnswerResult:
    """Answer `question` from the articles of `event` available at `cutoff` (interface v2).

    `interactive=True` (the web app) limits model calls to 2 attempts, 20 s each and 30 s for the whole
    answer; the default keeps the long retries the batch evaluation uses."""
    started = _now()
    question = question if isinstance(question, str) else str(question or "")
    limit = events.parse_cutoff(event, cutoff)
    try:
        con = _index.connect(event)
    except _index.IndexMissing:
        return _empty(NOTICE_NO_INDEX)
    try:
        steps = switches()
        key, when, notice = _locate(con, question, limit, event, steps)
        if notice is not None:
            return _empty(notice)
        with _lock_for(key):
            state = _load(key)
            if state is None:
                state, degraded = _run(con, question, limit, when, event,
                                       _Limits(started) if interactive else None, steps)
                if state["draft"] is not None and not degraded:
                    _store(key, state)
    finally:
        con.close()
    with _timed("checks"):
        return _finish(state, question, limit, when, event, guard)
