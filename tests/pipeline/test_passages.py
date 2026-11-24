"""Passages: exact substrings of the article, chosen around the search terms."""
from __future__ import annotations

from pipeline.passages import MAX_WORDS, SEPARATOR, opening, passage


def _filler(n: int, tag: str) -> str:
    return " ".join(f"{tag}{i}" for i in range(n)) + "."


def _long_text() -> str:
    return " ".join([
        _filler(30, "alpha"), "The Titanic struck an iceberg near midnight.", _filler(200, "beta"),
        _filler(200, "gamma"), "Survivors were taken aboard the Carpathia.", _filler(200, "delta"),
    ])


def test_passage_is_verbatim_substrings():
    text = _long_text()
    out = passage(text, ["titanic", "carpathia"], max_words=100)
    pieces = out.split(SEPARATOR)
    assert len(pieces) >= 2
    for piece in pieces:
        assert piece and piece in text
    assert "Titanic struck" in out and "Carpathia" in out


def test_passage_respects_word_limit():
    out = passage(_long_text(), ["titanic", "carpathia"], max_words=100)
    words = sum(len(p.split()) for p in out.split(SEPARATOR))
    assert words <= 100
    assert len(passage(_long_text(), ["zzz"]).split()) <= MAX_WORDS


def test_passage_short_text_unchanged():
    text = "The Titanic sank.  Hundreds were lost.\n\nMore later."
    assert passage(text, ["titanic"]) == text


def test_passage_merges_adjacent_spans():
    text = " ".join([_filler(30, "a"), "Titanic hit ice.", "Titanic took water.", _filler(30, "b")])
    out = passage(text, ["titanic"], max_words=12)
    assert SEPARATOR not in out
    assert out == "Titanic hit ice. Titanic took water."


def test_passage_no_hits_returns_opening():
    text = _filler(100, "w")
    out = passage(text, ["nothing"], max_words=20)
    assert text.startswith(out) and len(out.split()) <= 20 and out.split()[0] == "w0"


def test_passage_matches_inflections_by_stem():
    text = " ".join([_filler(50, "a"), "The ship was sinking fast.", _filler(50, "b")])
    out = passage(text, ["sink"], max_words=10)
    assert "sinking" in out


def test_opening_is_prefix():
    text = "  One two  three\nfour five six."
    out = opening(text, words=4)
    assert text.startswith(out) and out.split() == ["One", "two", "three", "four"]
    assert opening("short text") == "short text"
