"""Negation safety under OCR tolerance, as a matrix.

Every OCR-tolerant comparison in `pipeline.verify` (quote words, OCR splits joined, names, content words
under letter confusions) must refuse a match that adds, drops or flips a negation, whatever OCR damage
sits on either side: a negation word misread ("uot", "n0t") or split ("n ot"), a negating prefix misread
("nnsinkable", "lnactive") or split off ("un sinkable", "in active"), or the base word misread
("slnkable"). The positive matrix shows that the same tolerance still accepts real OCR damage that
leaves the negation alone.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from pipeline import verify
from pipeline.verify import MATCH_THRESHOLD, allowed_terms, check, quote_match, unsupported

UTC = timezone.utc
START = datetime(1912, 4, 10, tzinfo=UTC)
CUTOFF = datetime(1912, 4, 15, 23, 59, 59, tzinfo=UTC)
QUESTION = "What did the papers say?"


def _check(text: str, quote: str, passage: str):
    art = {"article_id": "a", "paper": "The Sun", "date": "1912-04-15", "available_at": "1912-04-15T23:59:59Z",
           "page": "1", "url": "", "state": "New York", "headline": "", "text": passage}
    allowed = allowed_terms(QUESTION, "the end of Monday, April 15, 1912", {"a": art}, START, CUTOFF)
    return check({"text": text, "quote": quote, "article_id": "a"}, passage, art, CUTOFF, {"a"}, QUESTION, allowed)[1]


# --- negation words ---------------------------------------------------------------------------------

# word: (OCR misreadings, split forms); "both" forms split each misreading where the split form splits.
NEGATION_WORDS = {
    "no": (["n0", "uo"], ["n o"]),
    "not": (["uot", "n0t", "nof"], ["n ot", "no t"]),
    "nor": (["n0r", "uor"], ["n or"]),
    "never": (["nevcr", "ncver"], ["ne ver"]),
    "none": (["n0ne", "noue"], ["no ne"]),
    "nothing": (["nothlng", "nothiug"], ["no thing", "noth ing"]),
    "nobody": (["nohody", "n0body"], ["no body"]),
    "nowhere": (["novhere", "n0where"], ["no where"]),
    "neither": (["nelther", "neithcr"], ["nei ther"]),
    "cannot": (["cann0t", "canuot"], ["can not"]),
}


def _negation_forms() -> list[tuple[str, str, str]]:
    out = []
    for word, (swaps, splits) in NEGATION_WORDS.items():
        out.append((word, "clean", word))
        out += [(word, "swap", s) for s in swaps]
        out += [(word, "split", s) for s in splits]
        for split in splits:
            cut = split.index(" ")
            out += [(word, "both", s[:cut] + " " + s[cut:]) for s in swaps]
    return out


NEGATION_FORMS = _negation_forms()
_NEG_IDS = [f"{w}-{d}-{f}" for w, d, f in NEGATION_FORMS]

PLAIN_PASSAGE = "the captain was able to reach the bridge in time"
PLAIN_SENTENCE = "The paper reported that the captain was able to reach the bridge."


def _negated(form: str) -> str:
    return f"the captain was {form} able to reach the bridge in time"


@pytest.mark.parametrize("word, damage, form", NEGATION_FORMS, ids=_NEG_IDS)
def test_negation_word_dropped_from_a_damaged_passage(word, damage, form):
    passage = _negated(form)
    # quote path: the plain quote never matches the negated passage
    assert quote_match(PLAIN_PASSAGE, passage) < MATCH_THRESHOLD
    # join path: the negation never joins its neighbour into a plain word ("uot able" is not "notable")
    assert quote_match(f"the captain was {word}able to reach the bridge in time", passage) < MATCH_THRESHOLD
    # polarity path: a verbatim OCR quote still carries the negation the sentence drops
    reason = unsupported(PLAIN_SENTENCE, passage, QUESTION, set(), quote=passage)
    assert reason == "drops a negation its source contains", reason
    assert _check(PLAIN_SENTENCE, passage, passage) == "drops a negation its source contains"
    # and a cleaned-up quote does not match the damaged passage unless the passage is clean
    if damage != "clean":
        assert _check(PLAIN_SENTENCE, _negated(word), passage) is not None


@pytest.mark.parametrize("word, damage, form", NEGATION_FORMS, ids=_NEG_IDS)
def test_negation_word_inserted_against_a_plain_passage(word, damage, form):
    # quote path: a quote holding the negation, damaged or not, never matches the plain passage
    assert quote_match(_negated(form), PLAIN_PASSAGE) < MATCH_THRESHOLD
    assert _check(f"The paper reported that the captain was {word} able to reach the bridge.",
                  _negated(form), PLAIN_PASSAGE) is not None
    # polarity path: the sentence's negation is not in its quote
    reason = unsupported(f"The paper reported that the captain was {word} able to reach the bridge.",
                         PLAIN_PASSAGE, QUESTION, set(), quote=PLAIN_PASSAGE)
    assert reason == "denies what its source says", reason


# --- negating prefixes --------------------------------------------------------------------------------

# prefix: (base word, OCR misreadings of the prefix, OCR misreadings of the base). The bases are long
# enough for the fuzzy name path (ratio >= 85 between the negated and the plain word) and the R9
# content-word path (6+ letters), so every path is exercised.
PREFIXES = {
    "un": ("sinkable", ["nn", "uu"], ["slnkable", "sinkab1e"]),
    "in": ("active", ["ln", "1n"], ["actlve", "act1ve"]),
    "im": ("possible", ["lm", "1m"], ["posslble", "p0ssible"]),
    "il": ("legitimate", ["ll", "1l"], ["legitlmate", "1egitimate"]),
    "ir": ("regular", ["lr", "1r"], ["regnlar", "regu1ar"]),
    "non": ("essential", ["n0n", "nou"], ["essentlal", "esseutial"]),
    "dis": ("connected", ["dls", "d1s"], ["connectcd", "c0nnected"]),
    "mis": ("represented", ["mls", "m1s"], ["represeuted", "represcnted"]),
    "a": ("typical", [], ["typlcal", "typ1cal"]),
    "an": ("aerobic", ["au"], ["aerohic", "aer0bic"]),
}
# Prefixes that are words of their own: "in active" is read as "in" + "active", so its content path
# supports "active" (never "inactive"); only the quote path tells "in active" from "active".
STANDALONE = frozenset("in im il ir non a an".split())


def _prefix_forms() -> list[tuple[str, str, str, str]]:
    """(prefix, damage, negated form, plain form): the negated word under damage on the prefix, the
    base, both, split off, and split with damage; the plain form under damage on the base."""
    out = []
    for prefix, (base, pswaps, bswaps) in PREFIXES.items():
        out.append((prefix, "clean", prefix + base, base))
        out += [(prefix, "swap-prefix", p + base, base) for p in pswaps]
        out += [(prefix, "swap-base", prefix + b, b) for b in bswaps]
        out += [(prefix, "swap-both", p + b, b) for p in pswaps for b in bswaps]
        out.append((prefix, "split", f"{prefix} {base}", base))
        out += [(prefix, "split-swap", f"{p} {base}", base) for p in pswaps]
        out += [(prefix, "split-swap", f"{prefix} {b}", b) for b in bswaps]
    return out


PREFIX_FORMS = _prefix_forms()
_PREFIX_IDS = [f"{p}-{d}-{n.replace(' ', '_')}" for p, d, n, _b in PREFIX_FORMS]


def _frame(word: str) -> str:
    return f"tonight the {word} captain was on the bridge in good time"


def _sentence(word: str) -> str:
    return f"The paper reported that the {word} captain was on the bridge."


@pytest.mark.parametrize("prefix, damage, negated, plain", PREFIX_FORMS, ids=_PREFIX_IDS)
def test_prefix_dropped_from_a_damaged_passage(prefix, damage, negated, plain):
    base = PREFIXES[prefix][0]
    passage = _frame(negated)
    # quote path: the plain word never matches the negated passage word, split or misread; "a typical"
    # split is the article and a word, which a quote may give as "typical"
    if not (damage.startswith("split") and prefix in ("a", "an")):
        assert quote_match(_frame(base), passage) < MATCH_THRESHOLD
        assert quote_match(_frame(plain), passage) < MATCH_THRESHOLD
    # nor may the quote start just after "un", "dis" or "mis" the OCR split off (a quote may start after
    # a standalone "in" or "a")
    if damage.startswith("split") and prefix not in STANDALONE:
        assert quote_match(_frame(base).split(" the ", 1)[1], passage) < MATCH_THRESHOLD
    # the two pieces join into the negated word only when the left piece is not a word of its own ("a",
    # "an", or "in"/"im"/"il"/"ir"/"non" under OCR); an OCR "au" is not the article, so "au aerobic"
    # joins like "au gust"
    if damage.startswith("split"):
        left = negated.split()[0]
        standalone = left in ("a", "an") or verify._skeleton(left) in verify._PREFIX_WORD_SKELETONS
        joins = quote_match(_frame(prefix + base), passage) >= MATCH_THRESHOLD
        assert joins == (not standalone), (prefix, negated)
    # a verbatim OCR quote still leaves the plain sentence unsupported (content and name paths), except
    # that a standalone prefix word split off is read as a word of its own
    if not (damage.startswith("split") and prefix in STANDALONE):
        reason = unsupported(_sentence(base), passage, QUESTION, set(), quote=passage)
        assert reason is not None and base in reason, reason
        reason = unsupported(_sentence(base.capitalize()), passage, QUESTION, set(), quote=passage)
        assert reason == f"names {base.capitalize()}, which its source does not mention", reason
        assert _check(_sentence(base), passage, passage) is not None


@pytest.mark.parametrize("prefix, damage, negated, plain", PREFIX_FORMS, ids=_PREFIX_IDS)
def test_prefix_inserted_against_a_damaged_passage(prefix, damage, negated, plain):
    base = PREFIXES[prefix][0]
    passage = _frame(plain)
    # quote path: the negated word, damaged, split or clean, never matches the plain passage word ("a
    # typical" in a quote is the article and a word, so it may stand for "typical")
    if not (damage.startswith("split") and prefix in ("a", "an")):
        assert quote_match(_frame(negated), passage) < MATCH_THRESHOLD
    assert quote_match(_frame(prefix + base), passage) < MATCH_THRESHOLD
    # content path (R9 letter confusions) and name path (fuzzy ratio, OCR skeleton) refuse the negated word
    word = prefix + base
    reason = unsupported(_sentence(word), passage, QUESTION, set(), quote=passage)
    assert reason is not None and word in reason, reason
    reason = unsupported(_sentence(word.capitalize()), passage, QUESTION, set(), quote=passage)
    assert reason == f"names {word.capitalize()}, which its source does not mention", reason
    assert _check(_sentence(word), passage, passage) is not None


# --- a prefix split off before a word itself split in two ("un sea worthy") ---------------------------

THREE_PIECE = [(p, a, b) for p in ("un", "nn", "uu") for a, b in (("sea", "worthy"), ("sink", "able"))] + \
              [(p, "con", "nected") for p in ("dis", "dls")] + [(p, "rep", "resented") for p in ("mis", "mls")]


@pytest.mark.parametrize("prefix, left, right", THREE_PIECE, ids=lambda s: s)
def test_three_piece_split_never_supports_the_plain_word(prefix, left, right):
    base = left + right
    passage = f"tonight the ship was {prefix} {left} {right} when she sailed from the port"
    assert quote_match(f"tonight the ship was {base} when she sailed", passage) < MATCH_THRESHOLD
    assert quote_match(f"{base} when she sailed from the port", passage) < MATCH_THRESHOLD
    reason = unsupported(f"The paper reported that the ship was {base}.", passage, QUESTION, set(), quote=passage)
    assert reason is not None and base in reason, reason
    reason = unsupported(f"The paper reported that the {base.capitalize()} ship sailed.", passage, QUESTION, set(),
                         quote=passage)
    assert reason == f"names {base.capitalize()}, which its source does not mention", reason
    assert _check(f"The paper reported that the ship was {base}.", passage, passage) is not None
    # the negated word is what the three pieces offer (with the prefix spelt as in the source)
    assert unsupported(f"The paper reported that the ship was {prefix}{base}.", passage, QUESTION, set(),
                       quote=passage) is None


# --- a joined source candidate on the name path ("on known" is not "Unknown") -------------------------

@pytest.mark.parametrize("left, right, name", [
    ("on", "known", "Unknown"), ("0n", "known", "Unknown"), ("ou", "known", "Unknown"),
    ("on", "sinkable", "Unsinkable"), ("0n", "sinkable", "Unsinkable"), ("on", "seen", "Unseen"),
    ("ln", "active", "Inactive"), ("in", "active", "Inactive"), ("a", "typical", "Atypical"),
    ("an", "aerobic", "Anaerobic"), ("ln", "capable", "Incapable"),
], ids=lambda s: s)
def test_joined_source_candidates_go_through_the_join_predicate(left, right, name):
    passage = f"tonight the {left} {right} sailor was buried at sea by the crew"
    sentence = f"The paper reported that the {name} sailor was buried at sea."
    assert unsupported(sentence, passage, QUESTION, set(), quote=passage) == \
        f"names {name}, which its source does not mention"
    # the same word in lowercase goes through the content path (stems and R9 letter confusions)
    reason = unsupported(sentence.replace(name, name.lower()), passage, QUESTION, set(), quote=passage)
    assert reason is not None and name.lower() in reason, reason
    assert _check(sentence, passage, passage) is not None
    assert quote_match(f"tonight the {name.lower()} sailor was buried at sea", passage) < MATCH_THRESHOLD


def test_split_prefix_still_joins_on_the_name_path():
    passage = "tonight the un known sailor was buried at sea by the crew"
    assert unsupported("The paper reported that the Unknown sailor was buried at sea.", passage, QUESTION, set(),
                       quote=passage) is None
    assert unsupported("The paper reported that the Known sailor was buried at sea.", passage, QUESTION, set(),
                       quote=passage) is not None


# --- contractions under OCR damage ("wasn t", "wasu't", "wasut") -------------------------------------

def _contraction_forms() -> list[tuple[str, str, str]]:
    out = []
    for word, base in verify._SPLIT_NT.items():
        if word == "cannot":
            continue
        head = word[:-1]                      # "wasn", "can", "won"
        damaged = head[:-1] + "u"             # the "n" read as "u": "wasu", "cau", "wou"
        for h, tag in ((head, "clean"), (damaged, "n-as-u")):
            out += [(base, f"{tag}-apostrophe", f"{h}'t"), (base, f"{tag}-space", f"{h} t"),
                    (base, f"{tag}-joined", f"{h}t"), (base, f"{tag}-spaced-apostrophe", f"{h} ' t")]
    return out


CONTRACTIONS = _contraction_forms()


@pytest.mark.parametrize("base, damage, form", CONTRACTIONS, ids=[f"{b}-{d}-{f}" for b, d, f in CONTRACTIONS])
def test_damaged_contraction_is_a_negation(base, damage, form):
    passage = f"the captain {form} able to reach the bridge in time"
    plain = f"the captain {base} able to reach the bridge in time"
    # dropped: the plain quote never matches, a verbatim quote still carries the negation
    assert quote_match(plain, passage) < MATCH_THRESHOLD
    assert unsupported(PLAIN_SENTENCE, passage, QUESTION, set(), quote=passage) == \
        "drops a negation its source contains"
    assert _check(PLAIN_SENTENCE, passage, passage) == "drops a negation its source contains"
    # inserted: the contraction never matches the plain passage, and the sentence's negation is unbacked
    assert quote_match(passage, plain) < MATCH_THRESHOLD
    negated = f"The paper reported that the captain {base} not able to reach the bridge."
    assert unsupported(negated, plain, QUESTION, set(), quote=plain) == "denies what its source says"
    # the clean contraction and its expansion still match the damaged passage
    assert quote_match(f"the captain {base}n't able to reach the bridge in time".replace("cann't", "can't")
                       .replace("willn't", "won't"), passage) >= MATCH_THRESHOLD
    assert quote_match(f"the captain {base} not able to reach the bridge in time", passage) >= MATCH_THRESHOLD
    assert unsupported(negated, passage, QUESTION, set(), quote=passage) is None


# --- a prefix joined by a hyphen ("non-combatant") ---------------------------------------------------

HYPHENATED = [(p, base, p + "-" + base) for p, (base, _ps, _bs) in PREFIXES.items()] + \
             [(p, base, d + "-" + base) for p, (base, pswaps, _bs) in PREFIXES.items() for d in pswaps]


@pytest.mark.parametrize("prefix, base, form", HYPHENATED, ids=lambda s: s)
def test_hyphenated_prefix_is_part_of_the_word(prefix, base, form):
    passage = _frame(form)
    # dropped: the plain word never matches, not even a quote starting at it
    assert quote_match(_frame(base), passage) < MATCH_THRESHOLD
    assert quote_match(_frame(base).split(" the ", 1)[1], passage) < MATCH_THRESHOLD
    reason = unsupported(_sentence(base), passage, QUESTION, set(), quote=passage)
    assert reason is not None and base in reason, reason
    assert unsupported(_sentence(base.capitalize()), passage, QUESTION, set(), quote=passage) == \
        f"names {base.capitalize()}, which its source does not mention"
    assert _check(_sentence(base), passage, passage) is not None
    # inserted: the hyphenated word never matches the plain word
    assert quote_match(passage, _frame(base)) < MATCH_THRESHOLD
    assert unsupported(_sentence(prefix + base), _frame(base), QUESTION, set(), quote=_frame(base)) is not None
    # positive: the hyphenated and the joined spelling are the same word (a misread prefix is OCR damage
    # the content check does not undo, so the sentence word is checked with the source's own spelling)
    assert quote_match(_frame(prefix + "-" + base), passage) >= MATCH_THRESHOLD
    assert quote_match(_frame(prefix + base), passage) >= MATCH_THRESHOLD
    joined = form.replace("-", "")
    assert unsupported(_sentence(joined), passage, QUESTION, set(), quote=passage) is None
    assert unsupported(_sentence(form), passage, QUESTION, set(), quote=passage) is None


def test_prefix_split_into_letters_never_supports_the_plain_word():
    passage = "tonight the u n sinkable ship sailed on from the port"
    assert quote_match("tonight the sinkable ship sailed on", passage) < MATCH_THRESHOLD
    assert quote_match("sinkable ship sailed on from the port", passage) < MATCH_THRESHOLD
    for text in ("The paper reported that the sinkable ship sailed.",
                 "The paper reported that the Sinkable ship sailed."):
        assert unsupported(text, passage, QUESTION, set(), quote=passage) is not None
    assert unsupported("The paper reported that the unsinkable ship sailed.", passage, QUESTION, set(),
                       quote=passage) is None


@pytest.mark.parametrize("quote, passage", [
    ("in August the ship sailed from the port", "in Au gust the ship sailed from the port"),
    ("the authority of the line said so tonight", "the au thority of the line said so tonight"),
    ("in autumn the ship sailed from the port", "in au tumn the ship sailed from the port"),
])
def test_ocr_au_still_joins(quote, passage):
    assert quote_match(quote, passage) >= MATCH_THRESHOLD


# --- the same tolerance still accepts real OCR damage -------------------------------------------------

STAR = ("The new liner Titanic, the largest steamship afloat, collided with ice off the Grand Banks. ALL PASSEN-\n"
        "GERS WERE TAKEN OFF in the boats and are on the way to New York aboard the Carpathla.")
FIGURES = "The Titanic sank on Monday morning after striking an iceberg. About 1,5OO lost their lives."


@pytest.mark.parametrize("quote, passage", [
    ("are on the way to New York aboard the Carpathia", STAR),
    ("ALL PASSENGERS WERE TAKEN OFF in the boats", STAR),
    ("About 1,500 lost their lives", FIGURES),
    ("the passengers were unharmed by the collision", "the passengers were un harmed by the collision"),
    ("the steamer was at a distance from the wreck", "the steamer was at a dis tance from the wreck"),
    ("the officers said they cannot give a list", "the officers said they can not give a list"),
    ("The Titanic having transferred all her passengers", "the tt. tanic. having transferred all her passengers"),
    ("the greatest ship ever built, is in collision with berg", "the greatest ship ever built, is in collision "
                                                                 "with a berg"),
    # negation on both sides, damaged on one
    ("the unsinkable captain was on the bridge", "the nnsinkable captain was on the bridge"),
    ("the unsinkable captain was on the bridge", "the un sinkable captain was on the bridge"),
    ("the unknown captain was on the bridge", "the un knowu captain was on the bridge"),
    ("the captain was not able to reach the bridge", "the captain was not able to reach the bridge"),
    ("there was nothing to report from the bridge", "there was no thing to report from the bridge"),
    ("the lookout did notice the deck before the watch", "the lookout did notice the deck before the watch"),
    ("the ship was in active service on the route", "the ship was in active service on the route"),
    ("a wireless station and an armed guard", "a wireless station and an armed guard"),
], ids=lambda s: s[:30] if isinstance(s, str) else s)
def test_real_ocr_damage_without_a_negation_change_still_matches(quote, passage):
    assert quote_match(quote, passage) >= MATCH_THRESHOLD


@pytest.mark.parametrize("sentence, passage", [
    ("The paper reported that the passengers were unharmed.", "the passengers were un harmed by the collision"),
    ("The paper reported that the steamer was at a distance from the wreck.",
     "the steamer was at a dis tance from the wreck"),
    ("The paper reported that the officers cannot give a list of the survivors.",
     "the officers said they can not give a list of the survivors"),
    ("The paper reported that a wireless from the Carpathia said the maid was aboard.",
     "a wireless from the carpathla said the maid was aboard"),
    ("The paper reported that the survivors were picked up from lifeboats.",
     "the survivors were picked up from iifeboats"),
    ("The paper reported that the Unsinkable captain was on the bridge.",
     "the nnsinkable captain was on the bridge that night"),
    ("The paper reported that the unsinkable captain was on the bridge.",
     "the un sinkable captain was on the bridge that night"),
    ("The paper reported that the ship was in active service.", "the ship was in active service on the route"),
    ("The paper reported that the captain was not able to reach the bridge.",
     "the captain was uot able to reach the bridge in time"),
], ids=lambda s: s[:40])
def test_real_ocr_damage_without_a_negation_change_still_supports(sentence, passage):
    assert unsupported(sentence, passage, QUESTION, set(), quote=passage) is None


def test_negation_sets_agree():
    assert "nowhere" in verify._NEGATIONS and "nowhere" in verify._TEXT_NEGATIONS
    assert verify._JOINED_NEGATIONS <= verify._NEGATIONS
    assert all(verify._negation_like(w) for w in ("uot", "n0t", "nof", "nevcr", "uo", "n0", "canuot"))
    assert not any(verify._negation_like(w) for w in ("on", "note", "non", "hot", "out", "nut", "now"))


# --- contractions spaced or damaged at the apostrophe ("could n't", "wasn'f", "was'nt") -----------------

def _spaced_contraction_forms() -> list[tuple[str, str, str]]:
    """The period's print convention of a split-off "n't" ("could n't", "ca n't"), and OCR or typesetting
    damage at the apostrophe: "t" read as "f", a space before or after it, the apostrophe before the "n",
    and apostrophe look-alikes (backtick, acute accent, modifier letter)."""
    out = []
    for word, base in verify._SPLIT_NT.items():
        if word == "cannot":
            continue
        head = word[:-1]                      # "wasn", "can", "won"
        stem_ = head[:-1]                     # "was", "ca", "wo"
        out += [(base, "spaced-n", f"{stem_} n't"), (base, "spaced-u", f"{stem_} u't"),
                (base, "t-as-f", f"{head}'f"), (base, "space-before", f"{head} 't"),
                (base, "space-after", f"{head}' t"), (base, "apostrophe-before-n", f"{stem_}'nt"),
                (base, "backtick", f"{head}`t"), (base, "acute", f"{head}´t"),
                (base, "modifier", f"{head}ʼt")]
    return out


SPACED_CONTRACTIONS = _spaced_contraction_forms()


@pytest.mark.parametrize("base, damage, form", SPACED_CONTRACTIONS,
                         ids=[f"{b}-{d}-{f}" for b, d, f in SPACED_CONTRACTIONS])
def test_spaced_or_damaged_contraction_is_a_negation(base, damage, form):
    passage = f"the captain {form} able to reach the bridge in time"
    assert verify._word_list(verify.normalise(form)) == [base, "not"]
    assert quote_match(f"the captain {base} able to reach the bridge in time", passage) < MATCH_THRESHOLD
    assert unsupported(PLAIN_SENTENCE, passage, QUESTION, set(), quote=passage) == \
        "drops a negation its source contains"
    assert _check(PLAIN_SENTENCE, passage, passage) == "drops a negation its source contains"
    negated = f"The paper reported that the captain {base} not able to reach the bridge."
    assert unsupported(negated, passage, QUESTION, set(), quote=passage) is None


def test_spaced_contraction_end_to_end():
    passage = "The ship could n't be saved, the officers said."
    assert _check("The ship was saved.", "The ship could n't be saved", passage) == \
        "drops a negation its source contains"
    assert _check("The ship could not be saved.", "The ship could n't be saved", passage) is None


@pytest.mark.parametrize("text, words", [
    ("can f. smith", ["can", "f", "smith"]),
    ("Capt. Wm. Smith can f", ["capt", "wm", "smith", "can", "f"]),
    ("the n t tables", ["the", "n", "t", "tables"]),
])
def test_letters_after_a_verb_are_not_a_contraction(text, words):
    assert verify._word_list(verify.normalise(text)) == words


# --- hyphen look-alikes and stops between a prefix and its word ("un–known", "un.known") ---------------

PREFIX_SEPARATORS = ["‐", "‑", "‒", "–", "­"]
# A stop, a comma or an em dash counts only after a prefix that is not a word of its own.
ABSORBING_SEPARATORS = [".", ",", "—"]
SEPARATED = [(p, sep) for p in PREFIXES if p not in ("a", "an") for sep in PREFIX_SEPARATORS] + \
            [(p, sep) for p in ("un", "dis", "mis", "nn", "dls", "m1s") for sep in ABSORBING_SEPARATORS]


def _base_of(prefix: str) -> str:
    plain = {"nn": "un", "dls": "dis", "m1s": "mis"}.get(prefix, prefix)
    return PREFIXES[plain][0]


@pytest.mark.parametrize("prefix, sep", SEPARATED, ids=[f"{p}-U+{ord(s):04X}" for p, s in SEPARATED])
def test_separated_prefix_never_offers_the_plain_word(prefix, sep):
    base = _base_of(prefix)
    passage = _frame(prefix + sep + base)
    assert base not in verify._source(passage).words
    assert quote_match(_frame(base), passage) < MATCH_THRESHOLD
    reason = unsupported(_sentence(base), passage, QUESTION, set(), quote=passage)
    assert reason is not None and base in reason, reason
    assert _check(_sentence(base), passage, passage) is not None
    # positive: the joined word is supported and its quote matches
    assert unsupported(_sentence(prefix + base), passage, QUESTION, set(), quote=passage) is None
    assert quote_match(_frame(prefix + base), passage) >= MATCH_THRESHOLD


@pytest.mark.parametrize("text", ["un–known", "un­known", "un.known", "un,known", "un—known"])
def test_separated_prefix_end_to_end(text):
    passage = f"The ship was {text} to them."
    assert "known" not in verify._source(passage).words
    assert _check("The ship was known to them.", f"The ship was {text} to them", passage) is not None


@pytest.mark.parametrize("quote, passage", [
    ("in the boat all night", "they sat in.the boat all night"),
    ("in the boat all night", "they sat in,the boat all night"),
    ("non the boat all night", "they sat non.the boat all night"),
    ("a the boat all night", "they sat a.the boat all night"),
])
def test_stop_after_a_word_prefix_is_a_stop(quote, passage):
    words = verify._source(passage).words
    assert "the" in words and "boat" in words
    assert quote_match(quote, passage) == 100


# --- capitalised names carrying a hyphenated negating prefix ("Non-Combatant", "un-American") ----------

@pytest.mark.parametrize("sentence, passage", [
    ("The Non-Combatant crew stayed.", "the non-combatant crew stayed"),
    ("Non-Union men were hired.", "non-union men were hired"),
    ("They were called un-American.", "they were called un-american"),
    ("They were called Un-American.", "they were called un-American"),
    ("The Dis-Connected wire was found.", "the dis-connected wire was found"),
])
def test_hyphenated_prefix_name_is_one_name(sentence, passage):
    assert unsupported(sentence, passage, "Who stayed?", set(), quote=passage) is None


@pytest.mark.parametrize("sentence, passage", [
    ("Union men were hired.", "non-union men were hired"),
    ("Non-Union men were hired.", "union men were hired"),
    ("The Combatant crew stayed.", "the non-combatant crew stayed"),
    ("The Non-Combatant crew stayed.", "the combatant crew stayed"),
    ("They were called un-American.", "they were called American"),
    ("They were called American.", "they were called un-American"),
])
def test_hyphenated_prefix_name_fails_closed(sentence, passage):
    assert unsupported(sentence, passage, "Who stayed?", set(), quote=passage) is not None


def test_hyphenated_prefix_name_list():
    assert verify.names("The Non-Combatant crew met Captain Smith.", "Who stayed?") == {"Non-Combatant", "Smith"}
    assert verify.names("Anglo-American papers said so.", "Who stayed?") == {"Anglo", "American"}
