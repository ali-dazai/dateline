"""Checks that decide whether an answer sentence is backed by its quote and its source.

A sentence is kept only if its quote is found, word by word, in the passage of an article that was
given to the model and was available at the cutoff, and if every number, name and content word of the
sentence is in that passage or is always allowed (the question's words, paper names and places, the
dates from the story's start to the cutoff, the moment text). The content-word check is what catches
paraphrased hindsight such as "went down with his ship" or "will file for bankruptcy". A sentence may
not add a negation its quote lacks or drop one it has; the question's own words are accepted only in
statements about what the papers do not yet report.

Quote matching tolerates OCR damage (line-break hyphens, words split in two, common letter and digit
confusions such as b/h, i/l, 0/O) but not an added, dropped or changed word: a quote with "not"
inserted, "unsafe" for "safe", or ellipsis parts stitched from scattered words fails even though its
character-level similarity stays high. Every OCR-tolerant comparison (quote words, joined splits, names,
content words) first asks one negation-safety predicate (see "negation safety" below), so the tolerance
never lets a negated form stand for an un-negated one, whatever the damage ("uot able" for "notable",
"actlve" for "Inactive", "un safe" for "safe").
"""
from __future__ import annotations

import bisect
import os
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from functools import lru_cache

from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein

from dateline.events import parse_cutoff
from pipeline.search import STOP_WORDS, stem, tokens

MIN_QUOTE_WORDS = 6
MATCH_THRESHOLD = int(os.getenv("DATELINE_MATCH_THRESHOLD", "90"))
WORD_RATIO = 80
NAME_RATIO = 85
CONTEXT_CHARS = 300

_MONTHS = ("january february march april may june july august september october november december")
_WEEKDAYS = "monday tuesday wednesday thursday friday saturday sunday"
_NUMBER_WORDS_TEXT = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
                      "fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy "
                      "eighty ninety hundred thousand million billion")

COMMON_WORDS: frozenset[str] = frozenset(f"""
a an the it its this that there these those he she they we i you his her their our my your him them us
some several many most all both no none neither either nothing nobody never without each every other
another such any few more less
on in at by as of for from to with without into onto upon over under after before during since until
about against between through across near off out up down
yet but and or nor so if when while where whether although though however meanwhile also then thus
still only even not is was were are be been has had have do did does can could will would shall should
may might must what which who whom whose why how here
according reports report news word papers paper dispatch dispatches despatch despatches
today tonight yesterday tomorrow morning afternoon evening night
mr mrs miss ms messrs capt captain col colonel gov governor gen general dr sir lord lady rev hon lieut
lieutenant sen senator rep president vice secretary mayor judge admiral commander prof jr sr st mt ft
north south east west northern southern eastern western northeast northwest southeast southwest
{_MONTHS} {_WEEKDAYS} {_NUMBER_WORDS_TEXT}
""".split())

ATTRIBUTION_WORDS: frozenset[str] = frozenset(
    "reported said says according papers paper dispatch despatch report reports stated claimed feared "
    "believed rumoured rumored word news yet still no not".split())

# Connectives and words about the reporting itself. They carry no event facts, and answer sentences use
# them constantly ("The Herald also reported..."), so they are not required to appear in the source.
_LINKING_WORDS = frozenset(
    "also however meanwhile although though whereas instead other others another either added told wrote "
    "printed carried headline headlines article articles story stories edition accounts account dispatches "
    "despatches telegram telegrams message messages statement announced quoted cited several many various "
    "numerous including described noted suggested indicated insisted explained mentioned writes "
    "whether likewise relayed "
    # hedges and directions that add no event fact ("roughly 1900", "towed toward Halifax", "might be")
    "roughly approximately toward towards might".split())

# Abbreviations in old newsprint and their full words; a source with either form supports both.
_ABBREVIATIONS = {
    "capt": ("captain",), "col": ("colonel",), "gov": ("governor",), "gen": ("general",),
    "lieut": ("lieutenant",), "lt": ("lieutenant",), "sen": ("senator",), "rep": ("representative",),
    "dr": ("doctor",), "prof": ("professor",), "rev": ("reverend",), "supt": ("superintendent",),
    "sec": ("secretary",), "secy": ("secretary",), "pres": ("president",), "co": ("company",),
    "corp": ("corporation",), "bros": ("brothers",), "st": ("saint", "street"), "mt": ("mount",),
    "ft": ("fort",), "jr": ("junior",), "sr": ("senior",), "dept": ("department",),
    "govt": ("government",), "assn": ("association",),
}

_NEGATIONS = frozenset("no not nor never none nothing nobody nowhere neither cannot".split())

_QUOTES = str.maketrans({"‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'", "`": "'", "´": "'", "\u02bc": "'",
                         "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"',
                         # hyphen look-alikes OCR and typesetting use between a prefix and its word
                         "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-",
                         "\u00ad": None})  # soft hyphen: invisible, so "un\u00adknown" is "unknown"
# A prefix that is not a word of its own ("un", "dis", "mis", or their OCR forms; see _absorbing_prefix)
# set off from its word by a stop, a comma or a dash read for a hyphen ("un.known", "dis,abled",
# "un\u2014known"): made the bare hyphen that _tokenise joins, so the plain word is never offered on its
# own. "in", "non", "a" and "an" are words, and a stop after them is a stop ("in.the"). One character
# for one, so offsets into the raw text stay exact.
_PREFIX_SEP = re.compile(r"\b([^\W_]{2,4})[.,\u2014](?=[^\W\d_]{3})")


def _prefix_sep(text: str) -> str:
    return _PREFIX_SEP.sub(lambda m: m.group(1) + "-" if _absorbing_prefix(m.group(1).lower()) else m.group(), text)


def _fold(text: str) -> str:
    """NFKC with quotes, apostrophes and hyphen look-alikes made plain. _QUOTES is applied before NFKC
    as well, as NFKC splits some of them ("\u00b4" becomes a space and a combining accent)."""
    return unicodedata.normalize("NFKC", text.translate(_QUOTES)).translate(_QUOTES)
_LINE_HYPHEN = re.compile(r"-\s*\n\s*")
_SPACE = re.compile(r"\s+")
_RUN = re.compile(r"[^\W_]+")
_ELLIPSIS = re.compile(r"\.(?:\s*\.){2,}")
_QUOTE_WORD = re.compile(r"[^\W_]+(?:[',.\-][^\W_]+)*")
# A name may carry a negating prefix joined by a hyphen ("Non-Combatant", "un-American"): one word, as
# _tokenise reads it, so the name check looks for "noncombatant" rather than "Non" and "Combatant".
_NAME = re.compile(r"(?:(?:un|in|im|il|ir|non|dis|mis)-(?=[^\W\d_]{3}))?[^\W\d_]+(?:'[^\W\d_]+)*", re.I)
_POSSESSIVE = re.compile(r"'s$", re.IGNORECASE)


_NT_BASE = {"wo": "will", "ca": "can", "sha": "shall"}
_SPLIT_NT = {w: _NT_BASE.get(w[:-2], w[:-2]) for w in (
    "couldnt wouldnt shouldnt werent wasnt isnt didnt dont doesnt wont cant hasnt hadnt arent havent "
    "mustnt neednt").split()}
_SPLIT_NT["cannot"] = "can"
# What may stand between a contraction's head and its split-off "t": an apostrophe, a space, or both
# (OCR and tokenised text also set the apostrophe off on both sides: "wasn ' t").
_NT_GAPS = ("'", " ", "' ", " '", " ' ")


def _tokenise(s: str) -> list[tuple[str, int, int]]:
    """Alphanumeric runs of a normalised string with their offsets; "n't" becomes a separate "not".

    "couldn't", "couldnt" and "cannot" all give ("could"/"can", "not"), so a negation is always its own
    word and must be matched exactly; so do their OCR forms, where the apostrophe was read as a space or
    lost and the "n" misread ("wasn t", "wasu't", "wasu t", "wasut", "dldnt"), judged by skeleton. A
    negating prefix joined to its word by a bare hyphen ("non-combatant", "un-named") is one word."""
    runs = list(_RUN.finditer(s))
    out: list[tuple[str, int, int]] = []
    k = 0
    while k < len(runs):
        m = runs[k]
        word = m.group()
        nxt = runs[k + 1] if k + 1 < len(runs) else None
        gap = s[m.end():nxt.start()] if nxt is not None else ""
        if nxt is not None and gap in _NT_GAPS and (nxt.group() == "t" or ("'" in gap and _skeleton(nxt.group()) == "t")):
            base = None
            if len(word) > 1 and word.endswith("n") and gap.strip() == "'":
                base = _NT_BASE.get(word[:-1], word[:-1])
            elif _skeleton(word) in _NT_HEAD_SKELETONS:
                base = _NT_HEAD_SKELETONS[_skeleton(word)]
            if base is not None:
                out += [(base, m.start(), m.end() - 1), ("not", m.end() - 1, nxt.end())]
                k += 2
                continue
        # A print convention of the period, and OCR: the "n" of "n't" set off from its verb ("could n't",
        # "did n't"), or the apostrophe before the "n" ("was'nt"): "n" + "t" is "not"; "nt" after an
        # apostrophe closes a contraction. OCR forms by skeleton ("u't", "wasu'f").
        if nxt is not None and len(word) == 1 and _skeleton(word) == "n" and _skeleton(nxt.group()) == "t" \
                and gap.strip("' ") == "" and "'" in gap:
            if out and out[-1][0] in _NT_BASE:  # "ca n't", "wo n't"
                out[-1] = (_NT_BASE[out[-1][0]], out[-1][1], out[-1][2])
            out.append(("not", m.start(), nxt.end()))
            k += 2
            continue
        if nxt is not None and _skeleton(nxt.group()) == "nt" and gap.strip() == "'":
            base = _SPLIT_NT.get(word + "nt") or _SPLIT_NT_SKELETONS.get(_skeleton(word + "nt"))
            if base is not None:
                out += [(base, m.start(), m.end()), ("not", nxt.start(), nxt.end())]
                k += 2
                continue
        if nxt is not None and gap == "-" and _prefix_form(word) and len(nxt.group()) >= 3:
            out.append((word + nxt.group(), m.start(), nxt.end()))
            k += 2
            continue
        split = _SPLIT_NT.get(word) or (_SPLIT_NT_SKELETONS.get(_skeleton(word)) if word not in _NEGATIONS else None)
        if split is not None:
            cut = m.end() - (3 if _skeleton(word) == _skeleton("cannot") else 2)
            out += [(split, m.start(), cut), ("not", cut, m.end())]
        else:
            out.append((word, m.start(), m.end()))
        k += 1
    return out


def _word_list(s: str) -> list[str]:
    return [w for w, _a, _b in _tokenise(s)]


def _str(value) -> str:
    return "" if value is None else str(value)


def _prepare(text: str) -> str:
    """normalise() without lowercasing."""
    text = _prefix_sep(_fold(_str(text)))
    text = _LINE_HYPHEN.sub("", text)
    return _SPACE.sub(" ", text).strip()


def normalise(text: str) -> str:
    """NFKC, curly quotes made straight, line-break hyphens joined, whitespace collapsed, lowercase."""
    return _prepare(text).lower()


# --- OCR-tolerant word comparison ------------------------------------------------------------------

_SKELETON = str.maketrans({"0": "o", "1": "l", "i": "l", "5": "s", "c": "e", "b": "h", "u": "n", "f": "t",
                           "w": "v"})


def _skeleton(word: str) -> str:
    """A word with letters OCR often confuses made equal (rn/m, b/h, c/e, i/l/1, 0/o, 5/s, u/n, f/t, v/w:
    "vater-tight", "vill")."""
    return word.replace("rn", "m").translate(_SKELETON)


# Contractions by OCR skeleton: "wasut" -> ("was", "not"); the head before a split-off "t" ("wasn",
# "wasu") -> its base.
_SPLIT_NT_SKELETONS = {_skeleton(w): b for w, b in _SPLIT_NT.items() if w != "cannot"}
_NT_HEAD_SKELETONS = {_skeleton(w[:-1]): b for w, b in _SPLIT_NT.items() if w != "cannot"}


def _has_digit(word: str) -> bool:
    return any(ch.isdigit() for ch in word)


# OCR misreadings of "the" in old newsprint ("tho latest reports", "tlie ship").
_OCR_WORDS = {"tho": "the", "tlie": "the", "ihe": "the", "tne": "the"}
# Words a quote may leave out or the passage may hold without the quote: "the" carries no fact and OCR
# often garbles it or glues it to the word before ("l5athe", "tn", "me survivors").
_LIGHT_WORDS = frozenset({"the"})


def _significant(word: str) -> bool:
    """Words that must be matched: 3+ characters (except "the"), any digit, or a negation such as "no"
    (an OCR-damaged one too: "uo", "n0")."""
    return (len(word) >= 3 and word not in _LIGHT_WORDS) or _has_digit(word) or _negation_like(word)


# --- negation safety -------------------------------------------------------------------------------
#
# Every OCR-tolerant comparison in this module (quote words, OCR splits joined, names, content words)
# asks _negation_flip / _join_negation_flip first. Two words may only match when their negation status
# is the same, judged on the plain words and again on their OCR skeletons: a negation word ("not",
# or its OCR forms "uot", "n0t", "nof") never matches a word that is not one, and a word is never
# matched with itself plus a negating prefix ("unsafe"/"safe", "Inactive"/"actlve"), where removing
# the prefix brings the two closer together. The tolerance that makes "Carpathla" stand for
# "Carpathia" is therefore never what makes a negated form stand for an un-negated one.

_NEG_PREFIXES = ("un", "in", "im", "il", "ir", "non", "dis", "mis", "a", "an")
_NEG_PREFIX_SKELETONS = tuple(dict.fromkeys(_skeleton(p) for p in _NEG_PREFIXES))
_NEGATION_SKELETONS = frozenset(_skeleton(w) for w in _NEGATIONS)


def _negation_status(word: str) -> tuple[bool, bool]:
    """(plain, skeleton): whether `word` is a negation word, and whether its OCR skeleton is one."""
    return word in _NEGATIONS, _skeleton(word) in _NEGATION_SKELETONS


def _negation_like(word: str) -> bool:
    """A negation word, exact or under OCR letter confusions ("not", "uot", "n0t", "nof", "nevcr")."""
    return word in _NEGATIONS or _skeleton(word) in _NEGATION_SKELETONS


def _prefix_negates(a: str, b: str) -> bool:
    """Whether `a` is `b` with a negating prefix, on the plain words or on their OCR skeletons:
    removing the prefix leaves a word closer to `b` (fewer edits), so the prefix is what separates them.
    "unsafe"/"safe", "unsatisfactory"/"satisfactorv" and "lnactive"/"active" are; "about"/"abont" is
    not ("bout" is further from "abont" than "about" is)."""
    for x, y, prefixes in ((a, b, _NEG_PREFIXES), (_skeleton(a), _skeleton(b), _NEG_PREFIX_SKELETONS)):
        base = None
        for p in prefixes:
            if len(x) > len(p) and x.startswith(p):
                if base is None:
                    base = Levenshtein.distance(x, y)
                if Levenshtein.distance(x[len(p):], y) < base:
                    return True
    return False


@lru_cache(maxsize=1 << 16)
def _negation_flip(a: str, b: str) -> bool:
    """Whether matching words `a` and `b` would add, drop or flip a negation: their negation status
    differs on the plain words or on their OCR skeletons, or one is the other with a negating prefix."""
    if _negation_status(a) != _negation_status(b):
        return True
    return _prefix_negates(a, b) or _prefix_negates(b, a)


# Standalone words that are also negating prefixes: "in active" is not "inactive", "a typical" is not
# "atypical". "un", "dis" and "mis" are not words, so "un harmed" and "dis tance" are OCR splits.
_PREFIX_WORDS = frozenset("in im il ir non".split())
_PREFIX_WORD_SKELETONS = frozenset(_skeleton(w) for w in _PREFIX_WORDS)
_ARTICLE_PREFIXES = frozenset("a an".split())
_ARTICLE_PREFIX_SKELETONS = frozenset(_skeleton(w) for w in _ARTICLE_PREFIXES)
# Joins of "in" + word that are not negations of the word.
_IN_WORDS = frozenset("into inside indeed instead information inform informed income inland inward insight "
                      "inlet input".split())
# Negation words that are also written as one word; the only joins a negation word may take part in.
_JOINED_NEGATIONS = frozenset("cannot nothing nobody nowhere".split())


def _prefix_form(word: str) -> bool:
    """Whether `word` is a negating prefix itself, exact or under OCR confusions ("un", "nn", "dis")."""
    return word in _NEG_PREFIXES or _skeleton(word) in _NEG_PREFIX_SKELETONS


def _absorbing_prefix(word: str) -> bool:
    """A negating prefix that is not a word of its own ("un", "dis", "mis", or their OCR forms), so a
    word after it is the rest of one word: "un safe" is "unsafe" split, never "un" and "safe"."""
    return _prefix_form(word) and _skeleton(word) not in _PREFIX_WORD_SKELETONS \
        and _skeleton(word) not in _ARTICLE_PREFIX_SKELETONS


def _significance(words: Iterable[str]) -> tuple[bool, ...]:
    """Which words of a list must be matched: the significant ones (see _significant), and short ones
    that form a negation word with a neighbour ("n o", "n ot") or are a negating prefix, whole or split
    into letters, before a word of 3+ letters ("un safe", "in active", "u n safe"), so neither side of an
    alignment can skip them. The articles
    "a" and "an" stay skippable: a quote may read "with berg" for "with a berg", and "a typical" is
    kept apart from "atypical" by the join rules instead."""
    words = list(words)
    out: list[bool] = []
    for i, w in enumerate(words):
        prev = words[i - 1] if i else ""
        nxt = words[i + 1] if i + 1 < len(words) else ""
        after = words[i + 2] if i + 2 < len(words) else ""
        out.append(_significant(w) or bool(prev and _negation_like(prev + w)) or bool(nxt and _negation_like(w + nxt))
                   or (len(nxt) >= 3 and _prefix_form(w) and w not in _ARTICLE_PREFIXES)
                   or (len(nxt) >= 3 and len(prev + w) <= 3 and _absorbing_prefix(prev + w))
                   or (len(after) >= 3 and len(w + nxt) <= 3 and _absorbing_prefix(w + nxt)))
    return tuple(out)


def _after_split_prefix(words: tuple[str, ...], segs: tuple[int, ...], j: int) -> bool:
    """Whether `words[j]` follows "un", "dis" or "mis" standing alone (or split into letters, "u n") in
    the same segment, so it is the rest of a negated word; or follows a short such rest ("worthy" in
    "un sea worthy"), so it may be the end of one."""
    if j > 0 and segs[j - 1] == segs[j] and _absorbing_prefix(words[j - 1]):
        return True
    if j > 1 and segs[j - 2] == segs[j] and len(words[j - 2] + words[j - 1]) <= 3 \
            and _absorbing_prefix(words[j - 2] + words[j - 1]):
        return True
    return j > 0 and segs[j - 1] == segs[j] and len(words[j - 1]) <= 4 and _after_split_prefix(words, segs, j - 1)


def _join_unsafe(left: str, right: str) -> bool:
    """Whether `left` and `right` may never be read as one word, whatever they are compared with: a
    piece is a negation word, exact or OCR-damaged ("not able", "uot able", "not ice"), unless the join
    is itself a negation word ("can not" -> "cannot"); a standalone prefix word, exact or OCR-damaged
    ("in", "ln", "im", "il", "ir", "non"), before the word it would negate ("in active"), except the
    listed "in" words ("in to" -> "into"); or "a"/"an" before a word of 3+ letters ("a typical", "an
    armed"; so "a board" never stands for "aboard"). An OCR "au" is left to the prefix rules, so "au
    gust" still joins into "august"."""
    joined = left + right
    if joined in _JOINED_NEGATIONS:
        return False
    if _negation_like(left) or _negation_like(right):
        return True
    if joined in _IN_WORDS:
        return False
    if left in _ARTICLE_PREFIXES and len(right) >= 3:
        return True
    return _skeleton(left) in _PREFIX_WORD_SKELETONS


def _negates_piece(word: str, right: str, joined: str) -> bool:
    """Whether `word` is `right` with a negating prefix, judged against the join: removing the prefix
    leaves a word no further from `right` than `word` is from the whole join ("unsuccessfully" is
    "successfully" negated; "unknown" is "knowu" negated; "atlantic" is not "lantic" negated)."""
    for x, y, j, prefixes in ((word, right, joined, _NEG_PREFIXES),
                              (_skeleton(word), _skeleton(right), _skeleton(joined), _NEG_PREFIX_SKELETONS)):
        base = None
        for p in prefixes:
            if len(x) > len(p) and x.startswith(p):
                if base is None:
                    base = Levenshtein.distance(x, j)
                if Levenshtein.distance(x[len(p):], y) <= base:
                    return True
    return False


def _join_negation_flip(word: str, left: str, right: str) -> bool:
    """Whether reading `left` + `right` as `word` would add, drop or flip a negation.

    Refused when the join is unsafe in itself (see _join_unsafe); when the negation-word status of the
    join (any piece, or the joined word) differs from the word's, plainly or on OCR skeletons; when
    only one side carries a negating prefix: the join does if `left` is a prefix form ("un harmed",
    "nn harmed"), the word does if it is `right` with a prefix ("unsuccessfully" against "on
    successfully"); or when the joined word and `word` are a negating-prefix pair."""
    if _join_unsafe(left, right):
        return True
    joined = left + right
    plain = any(x in _NEGATIONS for x in (left, right, joined))
    skel = any(_skeleton(x) in _NEGATION_SKELETONS for x in (left, right, joined))
    if (plain, skel) != _negation_status(word):
        return True
    if _prefix_form(left) != _negates_piece(word, right, joined):
        return True
    return _prefix_negates(word, joined) or _prefix_negates(joined, word)


@lru_cache(maxsize=1 << 16)
def _same(a: str, b: str) -> bool:
    """Whether quote word `a` and passage word `b` are the same word, allowing for OCR damage.

    Negations must be identical and a negating prefix is never OCR damage ("unsafe" is not "safe"; see
    _negation_flip); words with digits and words shorter than 4 letters must agree once OCR confusions
    are made equal ("two" is not "to"); longer words may also differ a little (rapidfuzz ratio >=
    WORD_RATIO), but words under 8 letters only by at most one letter in length."""
    if a == b or _OCR_WORDS.get(a, a) == _OCR_WORDS.get(b, b):
        return True
    if a in _NEGATIONS or b in _NEGATIONS or _negation_flip(a, b):
        return False
    if _skeleton(a) == _skeleton(b):
        return True
    if _has_digit(a) or _has_digit(b) or min(len(a), len(b)) < 4:
        return False
    if min(len(a), len(b)) < 8 and abs(len(a) - len(b)) > 1:
        return False
    return fuzz.ratio(a, b) >= WORD_RATIO


_JOIN_RATIO = 90
# OCR letter confusions are accepted between a sentence word and a source word, or in a name, only for
# words of at least this many letters; shorter ones collide with other words (found/bound, Hill/bill).
_OCR_MIN_LETTERS = 6
# A join of the same length may differ a little more: a word split by OCR with a letter misread
# ("tt. tanic" for "Titanic"). The equal length keeps a whole neighbouring word from being absorbed.
_JOIN_RATIO_SAME_LENGTH = 85


@lru_cache(maxsize=1 << 16)
def _same_joined(word: str, left: str, right: str) -> bool:
    """Whether `word` is `left` and `right` written as one word (an OCR split, or "life boats").

    Stricter than _same, so a whole neighbouring word cannot be absorbed ("no" + "cause" is not "cause",
    "all" + "passengers" is not "passengers"); a negation may only be joined exactly ("can" + "not"), and
    no join may add, drop or flip a negation (see _join_negation_flip)."""
    if _join_negation_flip(word, left, right):
        return False
    joined = left + right
    if word == joined or _skeleton(word) == _skeleton(joined):
        return True
    if _has_digit(word) or _has_digit(joined):
        return False
    ratio = fuzz.ratio(word, joined)
    return ratio >= _JOIN_RATIO or (ratio >= _JOIN_RATIO_SAME_LENGTH and len(word) >= 6
                                    and abs(len(word) - len(joined)) <= 1)


# --- pieces: normalised words with their offsets in the raw text ------------------------------------

@dataclass(frozen=True)
class _Pieces:
    words: tuple[str, ...]
    starts: tuple[int, ...]   # raw offset of each word's first character
    ends: tuple[int, ...]     # raw offset just after each word's last character
    segs: tuple[int, ...]     # segment number; segments are separated by ellipses (passage separators)
    sig: tuple[bool, ...]     # whether each word must be matched (see _significance)


@lru_cache(maxsize=256)
def _pieces(text: str) -> _Pieces:
    """Alphanumeric runs of the normalised text, each mapped back to its span in `text`.

    Normalisation is applied character by character (NFKC, straight quotes, lowercase) so every output
    character knows the raw character it came from; line-break hyphens are then dropped."""
    chars: list[str] = []
    where: list[int] = []
    for k, ch in enumerate(text):
        for c in _fold(ch).lower():
            chars.append(c)
            where.append(k)
    flat = _prefix_sep("".join(chars))
    keep = [True] * len(flat)
    for m in _LINE_HYPHEN.finditer(flat):
        for i in range(m.start(), m.end()):
            keep[i] = False
    s = "".join(c for c, k in zip(flat, keep) if k)
    idx = [w for w, k in zip(where, keep) if k]
    barriers = [m.start() for m in _ELLIPSIS.finditer(s)]
    words, starts, ends, segs = [], [], [], []
    for word, a, b in _tokenise(s):
        words.append(word)
        starts.append(idx[a])
        ends.append(idx[b - 1] + 1)
        segs.append(bisect.bisect_right(barriers, a))
    return _Pieces(tuple(words), tuple(starts), tuple(ends), tuple(segs), _significance(words))


def _align(quote: list[str], p: _Pieces, lo: int, hi: int) -> list[tuple[int, int]]:
    """Every span p.words[s:e] with lo <= s <= hi that the quote words align to, as (s, e) by end.

    Alignment, left to right: a quote word matches one passage word, or two adjacent passage words
    joined (a word OCR split in two), or two quote words joined match one passage word ("life boats" /
    "lifeboats"); joins use the stricter _same_joined. Quote words that are not significant (1-2
    letters, no digit, not a negation or part of one, not a negating prefix before a word; see
    _significance) may be left unmatched; passage words inside the span that are not significant may
    be skipped. Every significant word on either side must be matched, so an
    added, dropped or changed word fails. The span stays within one segment and never starts right
    after a negating prefix split off by OCR ("un sinkable"). Each state keeps its
    latest start, so each end comes with its tightest span; the search is deterministic."""
    n, m = len(quote), len(p.words)
    if n == 0 or lo >= m:
        return []
    words, segs = p.words, p.segs
    sig_q = _significance(quote)
    rows: list[dict[int, int]] = [{} for _ in range(n + 1)]
    # A span may not start right after "un", "dis" or "mis" standing alone: the word it starts with is
    # the rest of a negated word split by OCR ("un sinkable"), and the quote would drop the prefix.
    rows[0] = {j: j for j in range(lo, min(m - 1, hi) + 1) if not _after_split_prefix(words, segs, j)}

    def put(i: int, j: int, s: int) -> bool:
        if rows[i].get(j, -1) < s:
            new = j not in rows[i]
            rows[i][j] = s
            return new
        return False

    for i in range(n):
        row = rows[i]
        order = sorted(row)
        k = 0
        while k < len(order):
            j = order[k]
            k += 1
            s = row[j]
            if not sig_q[i]:
                put(i + 1, j, s)
            if j >= m or segs[j] != segs[s]:
                continue
            if _same(quote[i], words[j]):
                put(i + 1, j + 1, s)
            # Two passage pieces join unless the second is the rest of a negated word and the first is
            # not its whole prefix ("n" + "sinkable" in "u n sinkable" never gives "sinkable").
            if j + 1 < m and segs[j + 1] == segs[j] and not (
                    _after_split_prefix(words, segs, j + 1) and not _absorbing_prefix(words[j])) \
                    and _same_joined(quote[i], words[j], words[j + 1]):
                put(i + 1, j + 2, s)
            if i + 1 < n and _same_joined(words[j], quote[i], quote[i + 1]):
                put(i + 2, j + 1, s)
            if s < j and not p.sig[j] and put(i, j + 1, s):
                bisect.insort(order, j + 1, lo=k)
    return [(s, e) for e, s in sorted(rows[n].items()) if s < e]


def _chunks(quote: str) -> list[str]:
    """The quote's parts between ellipses, normalised; parts without a word are dropped."""
    return [c.strip() for c in _ELLIPSIS.split(normalise(quote)) if _RUN.search(c)]


_MAX_PARTS = 3
_MIN_PART_WORDS = 4
_MIN_PART_SIGNIFICANT = 3
_MAX_PART_GAP = 40


def _parts_allowed(chunks: list[str]) -> bool:
    """A quote with ellipses has at most 3 parts of at least 4 words (3 significant) each, so it cannot
    be stitched together from scattered words."""
    if len(chunks) == 1:
        return True
    if len(chunks) > _MAX_PARTS:
        return False
    return all(len(_QUOTE_WORD.findall(c)) >= _MIN_PART_WORDS
               and sum(map(_counts_in_part, _word_list(c))) >= _MIN_PART_SIGNIFICANT for c in chunks)


def _counts_in_part(word: str) -> bool:
    """Words that count towards a part's minimum: 3+ characters, any digit, or a negation."""
    return len(word) >= 3 or _has_digit(word) or _negation_like(word)


def _locate(chunks: list[str], text: str) -> list[tuple[int, int]] | None:
    """Raw (start, end) of each part in `text`, or None when the parts cannot all be found.

    Parts must appear in order, each starting at most _MAX_PART_GAP words after the previous one ends."""
    if not chunks or not _parts_allowed(chunks):
        return None
    p = _pieces(text)
    parts = [_word_list(c) for c in chunks]
    failed: set[tuple[int, int]] = set()

    def solve(k: int, lo: int, hi: int) -> list[tuple[int, int]] | None:
        if (k, lo) in failed:
            return None
        for s, e in _align(parts[k], p, lo, hi):
            if k == len(parts) - 1:
                return [(s, e)]
            rest = solve(k + 1, e, e + _MAX_PART_GAP)
            if rest is not None:
                return [(s, e)] + rest
        failed.add((k, lo))
        return None

    found = solve(0, 0, len(p.words))
    if found is None:
        return None
    return [(p.starts[s], p.ends[e - 1]) for s, e in found]


def quote_match(quote: str, passage: str) -> int:
    """How well the quote matches the passage, 0-100: the lowest score over the quote's parts.

    A part scores rapidfuzz partial_ratio on its words against the passage's words (each joined by single
    spaces, so OCR punctuation such as "titanic , g disaster" or "total =: toll" does not count). The
    score is capped at MATCH_THRESHOLD - 1 when the word-level alignment fails (see _align and _locate),
    so a quote passes only if all of its parts are found word by word, in order and close together."""
    chunks = _chunks(_str(quote))
    passage = _str(passage)
    if not chunks or not passage.strip():
        return 0
    target = " ".join(_pieces(passage).words)
    score = min(int(fuzz.partial_ratio(" ".join(_word_list(chunk)), target)) for chunk in chunks)
    if _locate(chunks, passage) is None:
        score = min(score, MATCH_THRESHOLD - 1)
    return max(0, score)


def _display(quote: str, text: str) -> tuple[str, str]:
    """(context, hit): up to CONTEXT_CHARS of `text` around the quote, and the quoted span itself.

    `hit` is an exact substring of `context`, both slices of the raw text. When the quote is not found
    word by word, context is the quote and hit is empty."""
    spans = _locate(_chunks(quote), text) if text else None
    if not spans:
        return quote, ""
    start, end = spans[0][0], spans[-1][1]
    if end - start > CONTEXT_CHARS:
        start, end = spans[0]
    if end - start > CONTEXT_CHARS:
        return quote, ""
    room = CONTEXT_CHARS - (end - start)
    a = max(0, start - room // 2)
    b = min(len(text), a + CONTEXT_CHARS)
    a = max(0, min(a, b - CONTEXT_CHARS))
    b = max(b, end)
    if a > 0:
        cut = re.search(r"\s", text[a:start])
        if cut:
            a += cut.end()
    if b < len(text):
        tail = text[end:b]
        cut = max(tail.rfind(" "), tail.rfind("\n"))
        if cut >= 0:
            b = end + cut
    return text[a:b], text[start:end]


# --- numbers -----------------------------------------------------------------------------------------

_UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * (i + 2) for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
_SCALES = {"thousand": 10 ** 3, "million": 10 ** 6, "billion": 10 ** 9}
_COUNTED = frozenset(
    "reports report papers paper dispatches dispatch despatches despatch accounts account".split())
_ONE_PRONOUN_BEFORE = frozenset("no any every some each the this that which".split())
_DIGITISH = re.compile(r"([0-9OolI]*[0-9][0-9OolI]*|[0-9OolI]+)(st|nd|rd|th|d|s|bn|m)?")
_DIGIT_MAP = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1"})
_CURRENCY = "$£€"


def _plain(value: Decimal) -> str:
    if value == value.to_integral_value():
        return str(int(value))
    return format(value.normalize(), "f")


def _numbers(text: str) -> tuple[list[str], set[str]]:
    """(numbers in order of appearance, face values of digit runs before any scale was applied)."""
    text = _LINE_HYPHEN.sub("", unicodedata.normalize("NFKC", _str(text)))
    toks = _join_split_number_words(text, [(m.group(), m.start(), m.end()) for m in _RUN.finditer(text)])
    lows = [t[0].lower() for t in toks]
    found: list[str] = []
    faces: set[str] = set()

    def add(value: str) -> None:
        if value not in found:
            found.append(value)

    i = 0
    while i < len(toks):
        word, start, end = toks[i]
        m = _DIGITISH.fullmatch(word)
        if m:
            groups, suffix, j = [m.group(1)], m.group(2), i + 1
            decimal = None
            while not suffix and j < len(toks):
                sep = text[toks[j - 1][2]:toks[j][1]]
                nxt = _DIGITISH.fullmatch(toks[j][0])
                if sep not in (",", ".", " ") or not nxt:
                    break
                group = nxt.group(1)
                if len(group) == 3 and (sep != " " or (_has_digit(group) and _has_digit(groups[-1]))):
                    groups.append(group)
                    suffix = nxt.group(2)
                    j += 1
                    continue
                if sep == "." and len(groups) == 1 and _has_digit(group) and len(group) != 3:
                    decimal = group
                    suffix = nxt.group(2)
                    j += 1
                break
            digits = "".join(groups).translate(_DIGIT_MAP)
            if any(ch.isdigit() for ch in "".join(groups)):
                face = str(int(digits)) if decimal is None else f"{int(digits)}.{decimal.translate(_DIGIT_MAP)}"
                faces.add(face)
                factor = 1
                if suffix == "bn":
                    factor = 10 ** 9
                elif suffix == "m" and start > 0 and text[start - 1] in _CURRENCY:
                    factor = 10 ** 6
                elif not suffix and j < len(lows) and lows[j] in _SCALES:
                    factor = _SCALES[lows[j]]
                    j += 1
                elif not suffix and j < len(lows) and lows[j] == "hundred":
                    factor = 100
                    j += 1
                add(face if factor == 1 else _plain(Decimal(face) * factor))
            i = j
            continue
        if _is_count_word(lows, i) or lows[i] == "hundred" or (lows[i] == "a" and _is_scale(lows, i + 1)):
            i = _number_words(lows, i, add)
            continue
        i += 1
    return found, faces


_NUMBER_WORD_SET = frozenset(list(_UNITS) + list(_TENS) + ["hundred"] + list(_SCALES))


_SPLIT_SCALES = frozenset({"hundred", "thousand", "million", "billion"})


def _join_split_number_words(text: str, toks: list[tuple[str, int, int]]) -> list[tuple[str, int, int]]:
    """Join two runs that OCR split out of one scale word ("hun dred" -> "hundred"). Only the scale words
    are joined, both parts must have 2+ letters and only spaces may separate them, so "it's even", "on e",
    "el even" and "for ty" state no number."""
    out: list[tuple[str, int, int]] = []
    i = 0
    while i < len(toks):
        if i + 1 < len(toks):
            a, b = toks[i][0].lower(), toks[i + 1][0].lower()
            gap = text[toks[i][2]:toks[i + 1][1]]
            if (a + b) in _SPLIT_SCALES and len(a) >= 2 and len(b) >= 2 and gap.strip() == "" \
                    and a not in _NUMBER_WORD_SET and b not in _NUMBER_WORD_SET:
                out.append((toks[i][0] + toks[i + 1][0], toks[i][1], toks[i + 1][2]))
                i += 2
                continue
        out.append(toks[i])
        i += 1
    return out


def _is_count_of_reports(lows: list[str], i: int, j: int) -> bool:
    """A single small number word counting reports or papers ("Two papers reported", "one of the papers")."""
    if j - i != 1 or lows[i] not in _UNITS:
        return False
    rest = lows[j:j + 3]
    if rest[:1] and rest[0] in _COUNTED:
        return True
    return rest[:2] == ["of", "the"] and len(rest) == 3 and rest[2] in _COUNTED


def _is_count_word(lows: list[str], i: int) -> bool:
    return i < len(lows) and (lows[i] in _UNITS or lows[i] in _TENS)


def _is_scale(lows: list[str], i: int) -> bool:
    return i < len(lows) and (lows[i] == "hundred" or lows[i] in _SCALES)


def _number_words(lows: list[str], i: int, add) -> int:
    """Parse number words starting at token i ("fifteen hundred", "two thousand two hundred"); next index."""
    total, current, last, j = 0, 0, "", i
    used = False
    while j < len(lows):
        w = lows[j]
        if w == "a" and not used and _is_scale(lows, j + 1):
            current, last, used = 1, "unit", True
        elif w in _UNITS:
            if last in ("unit", "tens") and not (last == "tens" and _UNITS[w] < 10):
                break
            current += _UNITS[w]
            last, used = "unit", True
        elif w in _TENS:
            if last in ("unit", "tens"):
                break
            current += _TENS[w]
            last, used = "tens", True
        elif w == "hundred":
            current = (current or 1) * 100
            last, used = "scale", True
        elif w in _SCALES:
            total += (current or 1) * _SCALES[w]
            current, last, used = 0, "scale", True
        elif w == "and" and last == "scale" and _is_count_word(lows, j + 1):
            pass  # "one hundred and fifty"
        else:
            break
        j += 1
    if j == i:
        return i + 1
    if _is_count_of_reports(lows, i, j):
        return j
    if j - i == 1 and lows[i] == "one" and (
            (i > 0 and lows[i - 1] in _ONE_PRONOUN_BEFORE) or (j < len(lows) and lows[j] in ("of", "another"))):
        return j
    add(str(total + current))
    return j


def numbers(text: str) -> set[str]:
    """Numbers stated in `text` as plain strings: digit runs (thousands separators dropped, OCR O/o -> 0 and
    l/I -> 1 inside digit runs, scale words applied: "3 million" -> "3000000", "$613bn" -> "613000000000")
    and number words combined ("fifteen hundred" -> "1500"). Counts of reports ("two papers reported")
    and the pronoun "one" ("no one", "one of") are not numbers."""
    return set(_numbers(text)[0])


# --- names and content words --------------------------------------------------------------------------

def _name_list(text: str, question: str) -> list[str]:
    qwords = set(tokens(_str(question)))
    out: list[str] = []
    for m in _NAME.finditer(_fold(_str(text))):
        word = _POSSESSIVE.sub("", m.group())
        if not word or not word[0].isupper():
            continue
        low = word.lower()
        if low in COMMON_WORDS or low in qwords or low.replace("'", "") in qwords or low.replace("-", "") in qwords:
            continue
        if word not in out:
            out.append(word)
    return out


def names(text: str, question: str) -> set[str]:
    """Capitalised words of `text` (sentence-initial ones included) that are not common words and not
    words of the question, in their original case."""
    return set(_name_list(text, question))


def content_words(text: str) -> set[str]:
    """Stems of the words of `text`, stop words removed."""
    return {s for t in tokens(_str(text)) if t not in STOP_WORDS and (s := stem(t))}


@dataclass(frozen=True)
class _Source:
    numbers: frozenset[str]
    words: tuple[str, ...]
    stems: frozenset[str]
    skeletons: frozenset[str]
    skeleton_words: tuple[str, ...]
    skeleton_originals: dict[str, tuple[str, ...]]
    single_by_length: dict[int, tuple[str, ...]]
    joins: dict[str, tuple[str, str]]   # joined word -> (left piece, right piece); a negating prefix is a piece


def _prefix_run(runs: list[str], i: int) -> int:
    """How many pieces from `runs[i]` form a negating prefix that is not a word of its own ("un", "dis",
    or split into letters, "u n"): 1, 2 or 0."""
    if _absorbing_prefix(runs[i]):
        return 1
    if i + 1 < len(runs) and len(runs[i]) + len(runs[i + 1]) <= 3 and _absorbing_prefix(runs[i] + runs[i + 1]):
        return 2
    return 0


def _source_words(runs: list[str]) -> tuple[list[str], dict[str, tuple[str, str]]]:
    """The words a run of source pieces offers on its own, and the joins of adjacent pieces as joined
    word -> (left, right). After "un", "dis" or "mis" standing alone (an OCR split), the next piece is
    only the rest of the joined word and every join that starts there carries the prefix: "un sea
    worthy" gives "un", "unsea" and "unseaworthy", never "sea" or "seaworthy"."""
    own: list[str] = []
    joins: dict[str, tuple[str, str]] = {}
    prefix_of: dict[int, str] = {}   # index of a piece that follows a split-off prefix -> that prefix
    inside: set[int] = set()         # indexes of the prefix pieces themselves
    i = 0
    while i < len(runs):
        k = _prefix_run(runs, i)
        if k:
            inside.update(range(i, i + k))
            if i + k < len(runs):
                prefix_of[i + k] = "".join(runs[i:i + k])
            i += k
        else:
            i += 1
    for j, w in enumerate(runs):
        if j not in prefix_of:
            own.append(w)
        if j + 1 >= len(runs):
            continue
        nxt = runs[j + 1]
        if j + 1 in prefix_of:
            joins[prefix_of[j + 1] + nxt] = (prefix_of[j + 1], nxt)
        elif j in prefix_of:
            joins[prefix_of[j] + w + nxt] = (prefix_of[j], w + nxt)
        elif j in inside or not _join_unsafe(w, nxt):
            if j not in inside:
                joins[w + nxt] = (w, nxt)
    return own, joins


@lru_cache(maxsize=512)
def _source(text: str) -> _Source:
    """What a source supports: its numbers, and its words with OCR-split neighbours joined and
    abbreviations expanded, as words (for names), stems and OCR skeletons (for content words)."""
    found, faces = _numbers(text)
    norm = normalise(text)
    words: list[str] = []
    single: set[str] = set()
    joins: dict[str, tuple[str, str]] = {}
    for chunk in re.split(r"[^\w\s'\-]+", norm):
        own, chunk_joins = _source_words(_word_list(chunk))
        words.extend(own)
        single.update(own)
        joins.update(chunk_joins)
    words.extend(joins)
    expanded = set(words)
    for abbr, full in _ABBREVIATIONS.items():
        if abbr in expanded or any(f in expanded for f in full):
            expanded.add(abbr)
            expanded.update(full)
    alpha = sorted(w for w in expanded if not w.isdigit())
    skeletons = frozenset(_skeleton(w) for w in alpha)
    originals: dict[str, list[str]] = {}
    for w in alpha:
        originals.setdefault(_skeleton(w), []).append(w)
    by_length: dict[int, list[str]] = {}
    for w in sorted(single):
        if len(w) >= _OCR_MIN_LETTERS and not w.isdigit():
            by_length.setdefault(len(w), []).append(w)
    return _Source(numbers=frozenset(found) | frozenset(faces), words=tuple(alpha),
                   stems=frozenset(s for w in alpha if (s := stem(w))),
                   skeletons=skeletons, skeleton_words=tuple(sorted(skeletons)),
                   skeleton_originals={k: tuple(v) for k, v in originals.items()},
                   single_by_length={k: tuple(v) for k, v in by_length.items()},
                   joins={k: v for k, v in joins.items() if k not in single})


# The OCR confusions accepted between a sentence word and one source word: the ones the model actually
# corrected in the tuning set ("iifeboats", "afioat", "promlnent", "milllonaires") and digit/letter ones.
# b/h, f/t, c/e and u/n are left out: they join real words (hanged/banged, firing/tiring).
_CONTENT_CONFUSIONS = ({"i", "l", "1"}, {"o", "0"})
_VOWELS = frozenset("aeiou")


def _ocr_variant(word: str, other: str) -> bool:
    """Whether `other` is `word` with OCR letter confusions only: same length, every difference within
    i/l/1 or o/0, and no i/l difference right after a vowel, where both spellings are real words
    (toiled/tolled, failing/falling)."""
    if len(word) != len(other) or word == other:
        return False
    for k, (a, b) in enumerate(zip(word, other)):
        if a == b:
            continue
        group = next((g for g in _CONTENT_CONFUSIONS if a in g and b in g), None)
        if group is None:
            return False
        if "i" in group and k > 0 and word[k - 1] in _VOWELS:
            return False
    return True


def _ocr_supported(word: str, src: _Source) -> bool:
    """Whether one source word is `word` with OCR letter confusions only (see _ocr_variant): both 6+
    letters, never a joined pair of source words and never a negating-prefix pair."""
    if len(word) < _OCR_MIN_LETTERS:
        return False
    return any(_ocr_variant(word, w) and not _negation_flip(word, w)
               for w in src.single_by_length.get(len(word), ()))


@lru_cache(maxsize=1)
def _support_stems() -> frozenset[str]:
    return frozenset(s for w in ATTRIBUTION_WORDS | _LINKING_WORDS if (s := stem(w)))


_NUMBER_WORDS = frozenset(_NUMBER_WORDS_TEXT.split())


_TEXT_NEGATIONS = frozenset("no not never none nobody nowhere neither nor without nothing".split())


def _text_negation(word: str) -> bool:
    """A negation in quoted or source text: a text negation word, or a negation word under OCR letter
    confusions ("uot", "n0t")."""
    return word in _TEXT_NEGATIONS or _negation_like(word)


def _backing_negations(words: list[str], ignore: frozenset[str] = frozenset()) -> bool:
    """Whether quoted or source words hold a negation, OCR-damaged ("uot") or split in two ("n ot")."""
    return any(_text_negation(w) and w not in ignore for w in words) \
        or any(_negation_like(a + b) for a, b in zip(words, words[1:]))


_ABSENCE_NOUNS = frozenset(
    "paper papers report reports dispatch dispatches despatch despatches word news".split())
_ABSENCE_VERBS = frozenset("report say mention confirm".split())
# "not yet" opens an absence statement only before a word about reporting or knowing ("not yet reported",
# "not yet known"); "has not yet sunk" states the outcome.
_NOT_YET_WORDS = frozenset(
    "report reported reports reporting say said says mention mentioned mentions confirm confirmed confirms "
    "state stated states known clear certain evident printed published".split())

# Past forms whose stems differ from the question's word ("Did the Titanic sink?" / "...sank").
_IRREGULAR = {
    "sank": "sink", "sunk": "sink", "sunken": "sink", "struck": "strike", "lost": "lose", "went": "go",
    "gone": "go", "found": "find", "saw": "see", "seen": "see", "ran": "run", "broke": "break",
    "broken": "break", "fell": "fall", "fallen": "fall", "left": "leave", "took": "take", "taken": "take",
    "came": "come", "began": "begin", "begun": "begin", "told": "tell", "said": "say", "knew": "know",
    "known": "know", "flew": "fly", "flown": "fly", "caught": "catch", "brought": "bring", "rose": "rise",
    "risen": "rise", "gave": "give", "given": "give", "held": "hold", "kept": "keep", "sold": "sell",
    "sent": "send", "stood": "stand", "won": "win", "froze": "freeze", "frozen": "freeze", "drew": "draw",
    "drawn": "draw", "threw": "throw", "thrown": "throw", "blew": "blow", "blown": "blow", "swam": "swim",
    "paid": "pay", "met": "meet", "bought": "buy", "sought": "seek", "fought": "fight", "wrote": "write",
    "written": "write", "hit": "hit", "made": "make", "heard": "hear", "died": "die", "dead": "die",
}


# Past forms of the same irregular verb, for sentences that change a passage's tense or form: a past form
# in a sentence ("fell", "sunk") is supported by a past form of that verb in the source ("fallen",
# "sank"), and a present form ("sinking", "finding") by any form. A past form is never supported by a
# present one: "is sinking" in a source does not support "sank". "dead" is a state, not a past form.
_PAST_FORMS: dict[str, frozenset[str]] = {}
for _form, _lemma in _IRREGULAR.items():
    if _form != _lemma and _form != "dead":
        _PAST_FORMS[_lemma] = _PAST_FORMS.get(_lemma, frozenset()) | {_form}
_LEMMA_BY_STEM = {stem(lemma): lemma for lemma in _PAST_FORMS}

# Derived words and the source phrases that support them, written as joined words because a source's
# adjacent words are also stored joined ("taken over" -> "takenover").
_DERIVATIONS = {"takeover": ("takeover", "takenover", "tookover", "takingover", "takesover"),
                "night": ("tonight",), "day": ("today", "yesterday")}


def _irregular_supported(word: str, src: "_Source") -> bool:
    lemma = _IRREGULAR.get(word)
    if lemma is not None and word in _PAST_FORMS.get(lemma, ()):
        return any(stem(f) in src.stems for f in _PAST_FORMS[lemma])
    lemma = _LEMMA_BY_STEM.get(stem(word))
    return lemma is not None and any(stem(f) in src.stems for f in _PAST_FORMS[lemma])


_CLAUSE_BREAK = re.compile(r",\s*(?:but|yet|though|although|while)\b|[;:]|—|--|[.!?](?=\s|$)", re.I)
_NOT_A_BREAK = frozenset(_ABBREVIATIONS) | frozenset("mr mrs ms messrs no nos vs".split())


def _clauses(s: str, toks: list[tuple[str, int, int]]) -> list[int]:
    """Clause number of each token of normalised text `s`. Clauses end at ", but", ", yet", ", though",
    ", although", ", while", ";", ":", a dash or a sentence end (not the period of "Capt." or "E.")."""
    breaks = []
    for m in _CLAUSE_BREAK.finditer(s):
        if m.group() == ".":
            before = re.search(r"([^\W_]+)$", s[:m.start()])
            if before and (len(before.group(1)) == 1 or before.group(1).lower() in _NOT_A_BREAK):
                continue
        breaks.append(m.start())
    return [bisect.bisect_right(breaks, start) for _w, start, _e in toks]


def _not_yet_word(after: list[str]) -> bool:
    """Whether the words after "not yet" start with a reporting or knowing word, "been" allowed before it."""
    if after[:1] == ["been"]:
        after = after[1:]
    return after[:1] != [] and after[0] in _NOT_YET_WORDS


def _absence_frames(words: list[str], clause: list[int]) -> list[int]:
    """Indexes of negations that open an absence statement about the reports themselves, within one
    clause: "no paper(s) / report(s) / dispatch(es) / word / news ... yet", "nothing ... yet", "not yet"
    before a reporting or knowing word ("not yet reported", "not yet known"), "do/does/did not (yet)
    report / say / mention / confirm"."""
    out: list[int] = []
    for i, w in enumerate(words):
        later = [x for x, c in zip(words[i + 1:], clause[i + 1:]) if c == clause[i]]
        if w == "no" and any(x in _ABSENCE_NOUNS for x in later[:3]) and "yet" in later:
            out.append(i)
        elif w == "nothing" and "yet" in later:
            out.append(i)
        elif w == "not" and later[:1] == ["yet"] and _not_yet_word(later[1:3]):
            out.append(i)
        elif w == "not" and i > 0 and clause[i - 1] == clause[i] and words[i - 1] in ("do", "does", "did"):
            verb = later[1:2] if later[:1] == ["yet"] else later[:1]
            if verb and verb[0] in _ABSENCE_VERBS:
                out.append(i)
    return out


# Reporting words may take "whether", "if" or "that"; factive words only "whether" or "if", since "it is
# not yet known that the Titanic sank" asserts the sinking.
_REPORT_WORDS = frozenset(
    "report reports reported say says said mention mentions mentioned confirm confirms confirmed".split())
_FACTIVE_WORDS = frozenset("known clear certain evident".split())
_COMPLEMENTISERS = frozenset("whether if that".split())
_OPEN_COMPLEMENTISERS = frozenset("whether if".split())
# The only words allowed between an absence frame and its reporting word: the frame's own nouns, "yet",
# auxiliaries and "in the papers" / "in any paper". A verb such as "denied" or "doubts" breaks the frame.
_FRAME_FILL = _ABSENCE_NOUNS | frozenset("nothing yet has have had do does did been is was".split())
_CONJUNCTIONS = frozenset("and but yet so or nor as because while though although since when".split())
_INNER = "'-"


def _frame_fill(words: list[str], k: int) -> bool:
    """Whether words[k] may stand between an absence frame and its reporting word."""
    w = words[k]
    if w in _FRAME_FILL:
        return True
    if w == "in":
        return words[k + 1:k + 2] in (["the"], ["any"]) and words[k + 2:k + 3] != [] and \
            words[k + 2] in _ABSENCE_NOUNS
    if w in ("the", "any"):
        return k > 0 and words[k - 1] == "in"
    return False


def _closed_tail(s: str) -> bool:
    """Whether `s` (the rest of a sentence) has no clause punctuation: no comma, semicolon, colon, dash,
    bracket, quotation mark or sentence end before its final full stop. Abbreviation periods ("Capt.",
    "St.") and apostrophes or hyphens inside words are allowed."""
    s = s.rstrip()
    if s[-1:] in (".", "!", "?"):
        s = s[:-1]
    for k, ch in enumerate(s):
        if ch.isalnum() or ch.isspace():
            continue
        inside = 0 < k < len(s) - 1 and s[k - 1].isalnum() and s[k + 1].isalnum()
        if ch in _INNER and inside:
            continue
        if ch == ".":
            before = re.search(r"([^\W_]+)$", s[:k])
            if before and (len(before.group(1)) == 1 or before.group(1).lower() in _NOT_A_BREAK):
                continue
        return False
    return True


def _complements(s: str, toks: list[tuple[str, int, int]], frames: list[int], clause: list[int]) -> list[int]:
    """Indexes of "whether" / "if" / "that" that directly follow an absence frame's reporting word or noun
    ("No paper yet reports whether ...", "not yet known whether ...", "No word yet whether ...") and open
    a complement that runs, closed, to the end of the sentence. Returns the complementiser indexes."""
    words = [w for w, _a, _b in toks]
    out: list[int] = []
    for f in frames:
        c = None
        for k in range(f + 1, min(len(words), f + 8)):
            if clause[k] != clause[f]:
                break
            w, nxt = words[k], words[k + 1] if k + 1 < len(words) else ""
            if w in _REPORT_WORDS and nxt in _COMPLEMENTISERS:
                c = k + 1
                break
            if w in _FACTIVE_WORDS:
                c = k + 1 if nxt in _OPEN_COMPLEMENTISERS else None
                break
            if w in ("word", "news"):
                after = k + 2 if nxt == "yet" else k + 1
                if after < len(words) and words[after] in _COMPLEMENTISERS:
                    c = after
                    break
            if not _frame_fill(words, k):
                break
        if c is None or clause[c] != clause[f]:
            continue
        if words[f] in ("no", "nothing") and "yet" not in words[f + 1:c]:
            continue
        if any(w in _CONJUNCTIONS for w in words[c + 1:]) or not _closed_tail(s[toks[c][2]:]):
            continue
        out.append(c)
    return out


_TITLES = frozenset(
    "mr mrs miss ms messrs capt captain col colonel gov governor gen general dr sir lord lady rev hon lieut "
    "lieutenant sen senator president secretary mayor judge admiral commander prof".split())
# A capitalised question word after one of these is a verb or an adjective, not a name ("did Drown",
# "is Dead", "all Drowned"); only a question with no casing information needs this rule.
_NOT_BEFORE_NAMES = frozenset(
    "do does did is are was were be been being has have had will would shall should can could may might "
    "must all some many few both several every each none".split())


def _name_candidate_ok(key: str, cand: str, src: "_Source") -> bool:
    """Whether source word `cand` (a word, or two pieces joined) may support the name `key`: the negation
    predicate for a word, or for a join with its pieces ("on known" never supports "Unknown")."""
    pieces = src.joins.get(cand)
    if pieces is not None:
        return not _join_negation_flip(key, *pieces)
    return not _negation_flip(key, cand)


def _name_supported(key: str, src: "_Source") -> bool:
    for cand, _score, _i in process.extract(key, src.words, scorer=fuzz.ratio, score_cutoff=NAME_RATIO, limit=None):
        if _name_candidate_ok(key, cand, src):
            return True
    return False


def _name_supported_ocr(key: str, src: "_Source") -> bool:
    """`_name_supported` on OCR skeletons; the negation guard sees the plain words (or join pieces)
    behind each skeleton ("Inactive" is never supported by "active" or "actlve")."""
    for cand, _score, _i in process.extract(_skeleton(key), src.skeleton_words, scorer=fuzz.ratio,
                                            score_cutoff=NAME_RATIO, limit=None):
        if any(_name_candidate_ok(key, w, src) for w in src.skeleton_originals.get(cand, ())):
            return True
    return False


def unsupported(text: str, source: str, question: str, allowed: set[str], quote: str = "") -> str | None:
    """The first reason `text` is not supported by `source`, or None.

    In order:
    1. a number the source does not state;
    2. a name (capitalised word) that matches no source word (case-insensitive, rapidfuzz ratio >=
       NAME_RATIO, so OCR damage is tolerated);
    3. a negation the quote (or, without a quote, the source) does not have, unless it opens an absence
       statement such as "no paper yet reports...";
    4. with a quote: dropping the quote's negation (a "without" phrase may be left out);
    5. content words whose stems are not in the source (OCR-split words joined, letter confusions and
       abbreviations such as "Capt." allowed for). Terms in `allowed`, attribution words such as
       "reported" and fact-free connectives such as "also" always pass; words only the question supplies
       pass only inside a closed "whether / if / that" complement of an absence statement ("No paper
       yet reports whether the Titanic sank", but not "..., but it did", "...reports it and the Titanic
       sank", "No paper has yet denied reports that..." or "It is not yet known that..."); a capitalised
       question word counts as a name where the question itself capitalises it after its first word, or,
       when the question has no capitals of its own, mid-clause; a title before such a name ("Captain
       Smith") passes."""
    text, question, quote = _str(text), _str(question), _str(quote)
    allowed = {str(a).lower() for a in (allowed or ())}
    src = _source(_str(source))

    for n in _numbers(text)[0]:
        if n not in src.numbers and n not in allowed:
            return f"states {n}, which its source does not mention"

    for name in _name_list(text, question):
        key = name.lower().replace("'", "").replace("-", "")
        # OCR letter confusions are made equal first, so "Carpaihla" still supports "Carpathia"; only for
        # names of 6+ letters, as short ones would match common words (Hill/bill, Ball/hall, Butt/hutt).
        if name.lower() in allowed or key in allowed or _name_supported(key, src) \
                or (len(key) >= _OCR_MIN_LETTERS and _name_supported_ocr(key, src)):
            continue
        return f"names {name}, which its source does not mention"

    cased = _prepare(text)
    norm = cased.lower()
    if len(norm) != len(cased):
        cased = norm
    toks = _tokenise(norm)
    words = [w for w, _a, _b in toks]
    capital = [cased[a0:a0 + 1].isupper() for _w, a0, _b in toks]
    clause = _clauses(cased, toks)
    frames = _absence_frames(words, clause)
    negations = [i for i, w in enumerate(words) if w in _TEXT_NEGATIONS]
    # The quote's (or source's) negations, OCR-damaged ones included ("uot", "n0"): a sentence may not
    # drop them, nor may a verbatim OCR quote hide one.
    backing = _word_list(normalise(quote or source))
    if any(i not in frames for i in negations) and not _backing_negations(backing):
        return "denies what its source says"
    # A sentence may leave out a "without" phrase, but not the quote's other negations.
    if quote and not negations and _backing_negations(backing, ignore=frozenset({"without"})):
        return "drops a negation its source contains"

    question_words = set(tokens(question))
    # The question's own casing. In a mixed-case question ("Is the captain dead?") a word after the first
    # is a name only if the question capitalises it; its first word, and every word of an all-lowercase or
    # ALL-CAPS question, carry no casing information and fall to the sentence-position rule.
    question_toks = [_POSSESSIVE.sub("", m.group()).replace("-", "") for m in _NAME.finditer(_fold(question))]
    question_names = {t.lower() for t in question_toks[1:] if t[0].isupper()}
    question_later = {t.lower() for t in question_toks[1:]} if question != question.lower() \
        and question != question.upper() else set()
    allowed_stems = {s for a in allowed if (s := stem(a))}
    question_stems = {s for t in question_words for s in (stem(t), stem(_IRREGULAR.get(t, t))) if s}
    support = _support_stems()
    # Words only the question supplies pass only inside a closed "whether / if / that" complement of an
    # absence statement: "No paper yet reports whether the Titanic sank", not "..., but it did".
    comps = _complements(cased, toks, frames, clause)
    opened = min(comps) if comps else None
    reporting = {c - 1 for c in comps} | {c - 2 for c in comps if words[c - 1] == "yet"}
    # The reporting or knowing word that opens a "not yet" frame ("not yet known", "not yet been reported")
    # is about the reporting itself, whatever follows it.
    for f in frames:
        if words[f] == "not" and words[f + 1:f + 2] == ["yet"]:
            j = f + 3 if words[f + 2:f + 3] == ["been"] else f + 2
            if words[j:j + 1] and words[j] in _NOT_YET_WORDS:
                reporting.add(j)

    def is_name(i: int) -> bool:
        """A capitalised word already handled by the name check: names not in the question were checked
        against the source. A word a mixed-case question has after its first word is a name only if the
        question capitalises it; a question word with no casing information counts as a name mid-clause
        unless it follows an auxiliary or a quantifier ("did Drown", "all Drowned"). "Drowned
        passengers..." is an ordinary word."""
        w = words[i]
        if not capital[i] or w in COMMON_WORDS:
            return False
        if w not in question_words:
            return True
        if w in question_later:
            return w in question_names
        starts = i == 0 or clause[i] != clause[i - 1] or words[i - 1] in _NOT_BEFORE_NAMES
        return not starts

    bad: list[str] = []
    for i, tok in enumerate(words):
        if len(tok) < 2 or tok in STOP_WORDS or tok in bad:
            continue
        if _has_digit(tok) or tok in _NUMBER_WORDS or tok in _TEXT_NEGATIONS or tok in allowed or is_name(i):
            continue
        if i in reporting:
            continue
        if tok in _TITLES and i + 1 < len(words) and words[i + 1] in question_words and is_name(i + 1):
            continue
        forms = (tok, *_ABBREVIATIONS.get(tok, ()))
        if any((s := stem(f)) in src.stems or s in allowed_stems or s in support or _ocr_supported(f, src)
               for f in forms):
            continue
        if _irregular_supported(tok, src) or any(d in src.words for d in _DERIVATIONS.get(tok, ())):
            continue
        from_question = any(f in question_words or stem(f) in question_stems
                            or stem(_IRREGULAR.get(f, f)) in question_stems for f in forms)
        if from_question and opened is not None and i > opened:
            continue
        bad.append(tok)
    if bad:
        return f"uses words its source does not contain: {', '.join(bad)}"
    return None


# --- always-allowed terms -----------------------------------------------------------------------------

def _words(text) -> set[str]:
    return set(_RUN.findall(normalise(_str(text))))


_DOMAIN = re.compile(r"(?:www\.)?(?:the)?([^\W\d_]+)\.(?:com|co\.uk|org|net)")


def _paper_words(paper) -> set[str]:
    """Words of a paper's name; a web name also gives its plain name ("theguardian.com" -> "guardian")."""
    out = _words(paper)
    m = _DOMAIN.fullmatch(normalise(_str(paper)))
    if m:
        out.add(m.group(1))
    return out


def _date_terms(day: date) -> set[str]:
    month = day.strftime("%B").lower()
    return {str(day.day), month, month[:3], day.strftime("%A").lower(), str(day.year)} | (
        {"sept"} if day.month == 9 else set())


def _as_date(value) -> date | None:
    try:
        return parse_cutoff("", _str(value).strip()[:10]).date()
    except ValueError:
        return None


def allowed_terms(question: str, moment: str, given_articles: dict[str, dict], start: datetime,
                  cutoff: datetime, extra: Iterable[str] = ()) -> set[str]:
    """Lowercase words and number strings a sentence may use without its source containing them.

    The question's numbers (its words are handled by unsupported, which accepts words only the question
    supplies only in absence statements); the words and numbers of the moment text and of `extra` (the
    engine passes the given items' times in display time); every given article's paper name, place and
    date; and every calendar date from `start` to `cutoff` as day of month, month name, weekday and year."""
    out = numbers(_str(question))
    for text in [moment, *extra]:
        out |= _words(text) | numbers(_str(text))
    for art in (given_articles or {}).values():
        art = art or {}
        out |= _paper_words(art.get("paper")) | _words(art.get("state"))
        day = _as_date(art.get("date"))
        if day is not None:
            out |= _date_terms(day)
    try:
        first, last = parse_cutoff("", start).date(), parse_cutoff("", cutoff).date()
    except (TypeError, ValueError):
        return out
    day = max(first, last - timedelta(days=400))
    while day <= last:
        out |= _date_terms(day)
        day += timedelta(days=1)
    return out


# --- the check ----------------------------------------------------------------------------------------

def _quote_words(quote: str) -> int:
    return len(_QUOTE_WORD.findall(normalise(quote)))


def _available(article: dict, limit: datetime) -> bool:
    stamp = _str(article.get("available_at")).strip() or _str(article.get("date")).strip()
    try:
        return parse_cutoff("", stamp) <= limit
    except (TypeError, ValueError):
        return False


def check(sentence: dict, passage: str, article: dict | None, cutoff: datetime, given: set[str],
          question: str, allowed: set[str]) -> tuple[dict, str | None]:
    """The contract sentence for `sentence` and the reason it must be removed (None when it stands).

    Reasons, checked in this order: the quote is too short; the quote is not found in the passage; the
    article was not given to the model; it was not available at the cutoff; the sentence states a number,
    names a name or uses content words its passage does not contain (see unsupported)."""
    sentence = sentence if isinstance(sentence, dict) else {}
    text, quote = _str(sentence.get("text")), _str(sentence.get("quote"))
    aid = _str(sentence.get("article_id"))
    passage, question = _str(passage), _str(question)
    limit = parse_cutoff("", cutoff)
    short = _quote_words(quote) < MIN_QUOTE_WORDS
    short_reason = f"quote shorter than {MIN_QUOTE_WORDS} words"

    if not isinstance(article, dict) or not aid or aid not in (given or set()):
        # `published` is the cutoff itself, so a flagged sentence never looks later than the cutoff (a
        # minute cutoff's date alone reads as the end of that day).
        out = {"text": text, "quote": quote, "article_id": "", "paper": "", "date": limit.date().isoformat(),
               "page": "", "url": "", "published": limit.isoformat(), "place": "", "match": 0, "context": quote,
               "hit": ""}
        return out, short_reason if short else "article was not among the sources given"

    match = quote_match(quote, passage)
    context, hit = _display(quote, _str(article.get("text")) or passage)
    out = {"text": text, "quote": quote, "article_id": aid, "paper": _str(article.get("paper")),
           "date": _str(article.get("date")), "page": _str(article.get("page")), "url": _str(article.get("url")),
           "published": _str(article.get("available_at")), "place": _str(article.get("state")),
           "match": match, "context": context, "hit": hit}
    if short:
        return out, short_reason
    if match < MATCH_THRESHOLD:
        return out, f"quote not found in the article (best match {match}%)"
    if not _available(article, limit):
        return out, "article was not published by the cutoff"
    return out, unsupported(text, passage, question, allowed, quote=quote)
