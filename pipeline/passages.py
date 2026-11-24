"""Passages: the parts of an article most about the search terms, as exact substrings of its text.

The answer step quotes articles verbatim, so a passage is never rebuilt from tokens: every piece is a
slice of the original string, and pieces are joined with SEPARATOR.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

from pipeline.search import stem

SEPARATOR = " … "
MAX_WORDS = 450
MAX_SPAN_WORDS = 60

_WORD = re.compile(r"\S+")
_RUN = re.compile(r"[^\W_]+")
_BREAK = re.compile(r"(?<=[.!?])\s+|\n\s*\n")


def opening(text: str, words: int = 60) -> str:
    """The first `words` words of `text`, as an exact prefix slice (the whole text if it is shorter)."""
    if words <= 0:
        return ""
    end = None
    for n, m in enumerate(_WORD.finditer(text), start=1):
        end = m.end()
        if n == words:
            return text[:end]
    return text if end is not None else text[:0]


def _spans(text: str) -> list[tuple[int, int, int]]:
    """(start, end, word count) of sentence-like spans, in order; long runs are cut every 60 words."""
    bounds: list[tuple[int, int]] = []
    pos = 0
    for m in _BREAK.finditer(text):
        bounds.append((pos, m.start()))
        pos = m.end()
    bounds.append((pos, len(text)))
    out: list[tuple[int, int, int]] = []
    for start, end in bounds:
        found = list(_WORD.finditer(text, start, end))
        for i in range(0, len(found), MAX_SPAN_WORDS):
            chunk = found[i:i + MAX_SPAN_WORDS]
            out.append((chunk[0].start(), chunk[-1].end(), len(chunk)))
    return out


def _hits(text: str, start: int, end: int, stems: set[str]) -> int:
    return sum(1 for w in _RUN.findall(text[start:end].lower()) if stem(w) in stems)


def passage(text: str, terms: Iterable[str], max_words: int = MAX_WORDS) -> str:
    """Up to `max_words` words of `text`, chosen around the words that match `terms` by stem.

    Spans with more term hits are taken first (earlier first on ties) while they fit the budget; the
    chosen spans come out in document order, adjacent ones as a single piece, joined with SEPARATOR.
    With no hits the opening of the text is returned."""
    spans = _spans(text)
    if sum(n for _s, _e, n in spans) <= max_words:
        return text
    stems = {s for term in terms for run in _RUN.findall(str(term).lower()) if (s := stem(run))}
    scored = [(_hits(text, s, e, stems), i) for i, (s, e, _n) in enumerate(spans)] if stems else []
    ranked = sorted((item for item in scored if item[0] > 0), key=lambda item: (-item[0], item[1]))
    if not ranked:
        return opening(text, max_words)
    chosen: list[int] = []
    used = 0
    for _score, i in ranked:
        n = spans[i][2]
        if used + n <= max_words:
            chosen.append(i)
            used += n
    if not chosen:
        return opening(text, max_words)
    pieces: list[list[int]] = []
    for i in sorted(chosen):
        if pieces and pieces[-1][-1] == i - 1:
            pieces[-1].append(i)
        else:
            pieces.append([i])
    return SEPARATOR.join(text[spans[p[0]][0]:spans[p[-1]][1]] for p in pieces)
