"""Search terms for a question: its own words at full weight, a few filtered additions at low weight.

The model may suggest synonyms and spellings, but it must not steer the search toward facts it knows
from later. Added terms therefore carry little weight, and any name, place or number that is not in
the question is dropped.
"""
from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping, Sequence

from dateline import llm
from pipeline import prompts
from pipeline.search import tokens

QUESTION_WEIGHT = 1.0
ADDED_WEIGHT = 0.3
MAX_ADDED = 8
TERMS_MAX_TOKENS = 200
EXPANSION_ENV = "DATELINE_TERM_EXPANSION"
_OFF = {"0", "false", "off", "no"}


def switch_on(name: str) -> bool:
    """An on/off setting read from the environment when called: on unless set to 0, false, off or no."""
    return os.environ.get(name, "1").strip().lower() not in _OFF


def expansion_on() -> bool:
    """Whether the model suggests added search terms (DATELINE_TERM_EXPANSION, default on)."""
    return switch_on(EXPANSION_ENV)


_PUNCT = " \t\n.,;:!?\"'()[]{}-"
_PIECE = re.compile(r"[^\W_]+")


def _capital_outside(raw: str, own: set[str]) -> bool:
    """True if the word is capitalised and the question does not use it (in any case)."""
    raw = raw.strip(_PUNCT)
    return bool(raw) and raw[0].isupper() and raw.lower() not in own


def question_terms(question: str) -> dict[str, float]:
    """The question's own search words, each at full weight."""
    return {word: QUESTION_WEIGHT for word in tokens(question)}


def filter_added(question: str, added: list[str]) -> list[str]:
    """Keep added words that cannot carry outside knowledge, split into single search words.

    Dropped: anything with a digit, and anything capitalised (a name or place) unless the question
    itself uses that word. Words already in the question and repeats are dropped too.
    """
    own = set(tokens(question))
    out: list[str] = []
    seen: set[str] = set()
    for term in added:
        if not isinstance(term, str):
            continue
        # Pieces are letter/digit runs in their original case, so a name glued to a word by a hyphen,
        # a slash or a curly quote ("lost-Halifax", "lost “Halifax”") is seen on its own.
        words = _PIECE.findall(term)
        # A capitalised first word that the question does not use is a name (or a sentence start):
        # drop the whole term, as the model gives no other way to tell them apart.
        if not words or _capital_outside(words[0], own):
            continue
        for raw in words:
            if any(ch.isdigit() for ch in raw) or _capital_outside(raw, own):
                continue
            for word in tokens(raw):
                if word in own or word in seen:
                    continue
                seen.add(word)
                out.append(word)
    return out


def _suggestions(question: str, moment: str, llm_opts: Mapping | None,
                 on_fallback: Callable[[str], None] | None = None) -> list[str]:
    try:
        reply = llm.complete_json(
            prompts.TERMS.format(question=question, moment=moment),
            max_tokens=TERMS_MAX_TOKENS, **(llm_opts or {}),
        )
    except ValueError:  # a reply arrived but is not JSON: a real, if useless, answer
        return []
    except Exception as err:  # noqa: BLE001 - the question's own words still search
        if on_fallback is not None:
            on_fallback(f"terms call failed: {type(err).__name__}")
        return []
    terms = reply.get("terms") if isinstance(reply, dict) else None
    if not isinstance(terms, list) or not all(isinstance(t, str) for t in terms):
        return []
    return terms


def search_terms(question: str, moment: str, story_terms: Sequence[str] = (),
                 llm_opts: Mapping | None = None,
                 on_fallback: Callable[[str], None] | None = None,
                 expand: bool | None = None) -> dict[str, float]:
    """Weighted search words: question words 1.0, then up to 8 accepted additions and the story's terms 0.3.

    `expand=False` skips the model call (no additions: question words and story terms only); None reads
    DATELINE_TERM_EXPANSION. `on_fallback` is called with a short reason when the model call raised and
    no additions were used (a reply that arrives but is malformed is not a fallback)."""
    weights = question_terms(question)
    if not weights:
        return {}
    if expansion_on() if expand is None else expand:
        for word in filter_added(question, _suggestions(question, moment, llm_opts, on_fallback))[:MAX_ADDED]:
            weights.setdefault(word, ADDED_WEIGHT)
    for term in story_terms:
        for word in tokens(term):
            weights.setdefault(word, ADDED_WEIGHT)
    return weights
