"""Checks: word-level quote matching with OCR tolerance, and the number, name and content-word checks."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from dateline.contract import validate_result
from pipeline import verify
from pipeline.passages import SEPARATOR
from pipeline.verify import (
    ATTRIBUTION_WORDS, COMMON_WORDS, CONTEXT_CHARS, MATCH_THRESHOLD, MIN_QUOTE_WORDS, NAME_RATIO, WORD_RATIO,
    allowed_terms, check, content_words, names, normalise, numbers, quote_match, unsupported,
)

UTC = timezone.utc
START = datetime(1912, 4, 10, 0, 0, 0, tzinfo=UTC)
CUTOFF = datetime(1912, 4, 15, 23, 59, 59, tzinfo=UTC)
MOMENT = "the end of Monday, April 15, 1912"
QUESTION = "Were the Titanic's passengers saved?"

# Realistic OCR snippets in the shape of the fixture corpus (tests/pipeline/conftest.py).
STAR = ("NEW YORK, April 15. The new liner Titanic, the largest steamship afloat, collided with ice "
        "off the Grand Banks. ALL PASSEN-\nGERS WERE TAKEN OFF in the boats and are on the way to "
        "New York aboard the Carpathla, says a dispatch to the White Star offices.")
HERALD = ("The White Star liner Titanic struck an iceberg on her maiden voyage, but wireless messages say "
          "that all pas- sengers are safe and that the big ship is being towed to Halifax.")
SALT_LAKE = ("The Titanic, the greatest ship ever built, is in collision with a berg. The "
             "White Star line says there is no cause for alarm, as the vessel is unsinkable.")
SUN = ("TITANIC  SAFE; ALL PASSEN-\nGERS TAKEN OFF SAFELY BY THE VIRGINIAN. The steamer is "
       "reported to be proceeding slowly to Halifax under her own steam.")
CREW = ("The officers of the Titanic are: Captain E. J. Smith, commander; H. T. Wilde, chief "
        "officer; W. M. Murdock, first officer.")
FIGURES = ("The Titanic sank on Monday morning after striking an iceberg. About 1,5OO lost their "
           "lives, according to the despatch.")
MISSOULIAN = ("Cape Race, N. F.  The Titanic is reported to have struck an iceberg. The steamer is "
              "believed to be unhurt and her people are in no danger.")
STITCH = ("The Titanic struck an iceberg near the Banks. All passengers are reported safe and the officers "
          "say the liner lost no time in sending wireless calls for aid.")
WORDS = ("Fifteen hundred lost their lives when the liner went down, the line's offices were told "
         "this evening.")


def _article(aid: str, text: str, paper: str = "The Evening Star", state: str = "District of Columbia",
             date: str = "1912-04-15", available_at: str = "1912-04-15T23:59:59Z", page: str = "1") -> dict:
    return {"article_id": aid, "paper": paper, "date": date, "available_at": available_at, "page": page,
            "url": f"https://chroniclingamerica.loc.gov/lccn/{aid}/{date}/ed-1/seq-{page}/",
            "scan_url": f"https://www.loc.gov/item/{aid}/{date}/ed-1/", "state": state, "headline": "",
            "text": text}


def _run(text: str, quote: str, art: dict, question: str = QUESTION, passage: str | None = None,
         cutoff: datetime = CUTOFF, extra_given: dict | None = None, given: set[str] | None = None):
    arts = {art["article_id"]: art, **(extra_given or {})}
    allowed = allowed_terms(question, MOMENT, arts, START, cutoff)
    sentence = {"text": text, "quote": quote, "article_id": art["article_id"]}
    return check(sentence, art["text"] if passage is None else passage, art, cutoff,
                 set(arts) if given is None else given, question, allowed)


# --- constants and helpers -------------------------------------------------------------------------

def test_constants():
    assert MIN_QUOTE_WORDS == 6 and MATCH_THRESHOLD == 90 and WORD_RATIO == 80
    assert NAME_RATIO == 85 and CONTEXT_CHARS == 300
    assert ATTRIBUTION_WORDS == frozenset(
        "reported said says according papers paper dispatch despatch report reports stated claimed "
        "feared believed rumoured rumored word news yet still no not".split())
    for word in ("the", "according", "april", "monday", "capt", "gov", "mrs", "north", "lady", "word"):
        assert word in COMMON_WORDS
    assert all(w == w.lower() for w in COMMON_WORDS)


def test_normalise():
    assert normalise("ALL PASSEN-\nGERS  WERE\n taken") == "all passengers were taken"
    assert normalise("“the ﬁrst” liner’s") == '"the first" liner\'s'
    assert normalise("  a\t\tb  ") == "a b"


def test_numbers_digits_words_and_ocr():
    assert numbers("About 1,500 died") == {"1500"}
    assert numbers("About 1,5OO died") == {"1500"}
    assert numbers("l,500 lost") == {"1500"}
    assert numbers("fifteen hundred lost") == {"1500"}
    assert numbers("two thousand two hundred souls") == {"2200"}
    assert numbers("one hundred and fifty boats") == {"150"}
    assert numbers("April 15, 1912") == {"15", "1912"}
    assert numbers("sank at 2.20 o'clock") == {"2.20"}
    assert numbers("debts of $613bn") == {"613000000000"}
    assert numbers("3 million dollars") == {"3000000"}
    assert numbers("twenty-two hours") == {"22"}
    assert numbers("No numbers here, I said.") == set()


def test_numbers_ignore_counted_reports():
    assert numbers("Two papers reported it") == set()
    assert numbers("One report said so, and several dispatches agreed") == set()
    assert numbers("two accounts differ") == set()
    assert numbers("two boats were lowered") == {"2"}
    assert numbers("no one knows; one of the boats") == set()


def test_names_filter_common_and_question_words():
    found = names("The Herald said Captain Smith of the Titanic reached Halifax on Monday.", QUESTION)
    assert found == {"Herald", "Smith", "Halifax"}
    assert names("Survivors landed. They said so.", "") == {"Survivors"}


def test_content_words_are_stems_without_stop_words():
    assert content_words("The passengers were taken off safely") == {"passeng", "taken", "safe"}


def test_allowed_terms_cover_question_moment_papers_dates_and_extra():
    arts = {"a": _article("a", STAR), "b": _article("b", SUN, paper="The Sun", state="New York")}
    allowed = allowed_terms(QUESTION, MOMENT, arts, START, CUTOFF, extra=["10:05 a.m. EDT"])
    assert "titanic" not in allowed and "passengers" not in allowed  # question words: see unsupported()
    assert "1500" in allowed_terms("Were 1,500 lost?", MOMENT, arts, START, CUTOFF)
    for term in ("end", "monday", "evening", "star", "sun", "new", "york",
                 "district", "columbia", "april", "1912", "10", "11", "15", "wednesday", "5", "edt"):
        assert term in allowed, term
    assert "16" not in allowed and "tuesday" not in allowed  # April 16, 1912 was a Tuesday


# --- quote matching -------------------------------------------------------------------------------

def test_exact_quote_passes():
    quote = "The Titanic, the greatest ship ever built, is in collision with a berg."
    assert quote_match(quote, SALT_LAKE) == 100


def test_ocr_damaged_quote_passes():
    passage = "TITANIC  SAFE; ALL PASSEN-\nGERS TAKEN OFF BY THE VIRGINIAN"
    assert quote_match("Titanic safe; all passengers taken off by the Virginian", passage) >= MATCH_THRESHOLD


def test_ocr_split_word_and_damaged_letters_pass():
    assert quote_match("wireless messages say that all passengers are safe", HERALD) >= MATCH_THRESHOLD
    assert quote_match("are on the way to New York aboard the Carpathia", STAR) >= MATCH_THRESHOLD
    damaged = "The steamsbip Titanlc was reported safe at Halifax this evening."
    assert quote_match("The steamship Titanic was reported safe at Halifax", damaged) >= MATCH_THRESHOLD


def test_two_quote_words_may_match_one_passage_word():
    passage = "The women were put into the lifeboats and lowered away before the ship settled."
    assert quote_match("the women were put into the life boats and lowered away", passage) >= MATCH_THRESHOLD


def test_ligature_and_curly_quotes_pass():
    passage = "The ﬁrst wireless said: “All passengers safe,” and the liner’s officers agreed."
    assert quote_match('The first wireless said: "All passengers safe," and the liner\'s officers', passage) \
        >= MATCH_THRESHOLD


def test_negation_inserted_in_quote_fails():
    quote = "says there is not cause for alarm, as the vessel is unsinkable"
    assert quote_match(quote, SALT_LAKE) < MATCH_THRESHOLD
    assert quote_match("the vessel is not unsinkable, the White Star line says", SALT_LAKE) < MATCH_THRESHOLD


def test_dropped_negation_or_word_fails():
    assert quote_match("White Star line says there is cause for alarm", SALT_LAKE) < MATCH_THRESHOLD
    assert quote_match("the greatest ship ever built, is in collision with berg", SALT_LAKE) >= MATCH_THRESHOLD
    assert quote_match("The Titanic, the greatest ship built, is in collision", SALT_LAKE) < MATCH_THRESHOLD


def test_changed_word_or_digit_fails():
    assert quote_match("The Titanic, the greatest ship ever built, is in collision with a bank", SALT_LAKE) \
        < MATCH_THRESHOLD
    assert quote_match("after striking an iceberg. About 1,600 lost their lives", FIGURES) < MATCH_THRESHOLD
    assert quote_match("after striking an iceberg. About 1,500 lost their lives", FIGURES) >= MATCH_THRESHOLD


def test_quote_longer_than_passage_fails():
    assert quote_match("The Titanic sank with all hands lost at sea", "The Titanic sank") < MATCH_THRESHOLD


def test_ellipsis_quote_parts_checked():
    good = "The Titanic, the greatest ship ever built … the vessel is unsinkable"
    assert quote_match(good, SALT_LAKE) >= MATCH_THRESHOLD
    assert quote_match(good.replace("…", "..."), SALT_LAKE) >= MATCH_THRESHOLD
    bad = "The Titanic, the greatest ship ever built … the vessel has sunk"
    assert quote_match(bad, SALT_LAKE) < MATCH_THRESHOLD
    out_of_order = "the vessel is unsinkable … The Titanic, the greatest ship ever built"
    assert quote_match(out_of_order, SALT_LAKE) < MATCH_THRESHOLD


def test_quote_may_not_cross_a_passage_separator():
    passage = "The steamer is believed to be unhurt" + SEPARATOR + "her people are in no danger."
    assert quote_match("The steamer is believed to be unhurt", passage) >= MATCH_THRESHOLD
    assert quote_match("believed to be unhurt her people are in no danger", passage) < MATCH_THRESHOLD


# --- check(): quote, source and cutoff ------------------------------------------------------------

def test_short_quote_removed():
    _, reason = _run("The Salt Lake Herald reported a collision.", "in collision with a berg",
                     _article("s", SALT_LAKE))
    assert reason == "quote shorter than 6 words"


def test_quote_not_found_reports_best_match():
    out, reason = _run("The Herald reported the ship sank.", "the great liner sank at sea this morning",
                       _article("s", SALT_LAKE))
    assert reason == f"quote not found in the article (best match {out['match']}%)"
    assert out["match"] < MATCH_THRESHOLD


def test_unknown_article_removed():
    art = _article("s", SALT_LAKE)
    _, reason = _run("The Herald said the vessel is unsinkable.", "there is no cause for alarm, as the vessel",
                     art, given={"other"})
    assert reason == "article was not among the sources given"
    sentence = {"text": "x", "quote": "there is no cause for alarm, as the vessel", "article_id": "ghost"}
    _, reason = check(sentence, "", None, CUTOFF, {"s"}, QUESTION, set())
    assert reason == "article was not among the sources given"


def test_post_cutoff_article_removed():
    art = _article("late", SALT_LAKE, date="1912-04-16", available_at="1912-04-16T23:59:59Z")
    _, reason = _run("The Salt Lake Herald said the vessel is unsinkable.",
                     "there is no cause for alarm, as the vessel is unsinkable", art)
    assert reason == "article was not published by the cutoff"


def test_reason_order_is_stable():
    art = _article("s", SALT_LAKE)
    # short quote beats unknown article
    _, reason = check({"text": "x", "quote": "no cause for alarm", "article_id": "ghost"}, "", None, CUTOFF,
                      set(), QUESTION, set())
    assert reason == "quote shorter than 6 words"
    # quote not found beats post-cutoff
    late = _article("late", SALT_LAKE, available_at="1912-04-16T23:59:59Z")
    _, reason = _run("x", "the great liner sank at sea this morning", late)
    assert reason.startswith("quote not found in the article")
    # post-cutoff beats an unsupported number
    _, reason = _run("About 1,500 died.", "there is no cause for alarm, as the vessel is unsinkable", late)
    assert reason == "article was not published by the cutoff"
    # number before name before content words
    _, reason = _run("Captain Smith said 1,500 drowned.", "there is no cause for alarm, as the vessel", art)
    assert reason == "states 1500, which its source does not mention"
    _, reason = _run("Captain Smith drowned.", "there is no cause for alarm, as the vessel", art)
    assert reason == "names Smith, which its source does not mention"


# --- check(): numbers, names, words ---------------------------------------------------------------

def test_number_not_in_source_removed():
    _, reason = _run("About 1,500 died, the Herald said.", "there is no cause for alarm, as the vessel",
                     _article("s", SALT_LAKE))
    assert reason == "states 1500, which its source does not mention"


def test_number_words_match_digits():
    art = _article("w", WORDS)
    _, reason = _run("The Evening Star reported that 1,500 lost their lives.",
                     "Fifteen hundred lost their lives when the liner went down", art)
    assert reason is None


def test_ocr_digits_match():
    art = _article("f", FIGURES)
    _, reason = _run("The Evening Star reported that about 1,500 lost their lives.",
                     "after striking an iceberg. About 1,5OO lost their lives", art)
    assert reason is None
    _, reason = _run("The Evening Star reported that about 1,500 lost their lives.",
                     "after striking an iceberg. About 1,500 lost their lives", art)
    assert reason is None


def test_name_not_in_source_removed():
    _, reason = _run("Captain Smith went down, the Herald said.", "there is no cause for alarm, as the vessel",
                     _article("s", SALT_LAKE))
    assert reason == "names Smith, which its source does not mention"


def test_name_in_question_allowed():
    question = "What did the papers say about Captain Smith?"
    art = _article("s", SALT_LAKE, paper="The Salt Lake Herald", state="Utah")
    _, reason = _run("The Salt Lake Herald said Smith's vessel is unsinkable.",
                     "The White Star line says there is no cause for alarm", art, question=question)
    assert reason == "drops a negation its source contains"  # the quote's "no" must be kept
    _, reason = _run("The Salt Lake Herald said Smith's vessel is unsinkable.",
                     "The Titanic, the greatest ship ever built, is in collision", art, question=question)
    assert reason is None


def test_ocr_name_matches():
    _, reason = _run("The Evening Star reported that the passengers are on the way to New York aboard the "
                     "Carpathia.", "are on the way to New York aboard the Carpathia", _article("e", STAR))
    assert reason is None


def test_paper_names_and_dates_allowed():
    sun = _article("sun", HERALD, paper="The Sun", state="New York")
    _, reason = _run("On April 15 the New York Sun reported that the Titanic struck an iceberg and that all "
                     "passengers are safe.", "wireless messages say that all pas- sengers are safe", sun)
    assert reason is None


def test_year_and_day_numbers_allowed():
    sun = _article("sun", HERALD, paper="The Sun", state="New York")
    _, reason = _run("By the end of Monday, April 15, 1912, the Sun reported that all passengers are safe.",
                     "wireless messages say that all pas- sengers are safe", sun)
    assert reason is None


def test_counting_attribution_allowed():
    sun = _article("sun", HERALD, paper="The Sun", state="New York")
    star = _article("e", STAR)
    _, reason = _run("Two papers reported that the Titanic struck an iceberg.",
                     "The White Star liner Titanic struck an iceberg on her maiden voyage", sun,
                     extra_given={"e": star})
    assert reason is None


def test_honest_paraphrase_kept():
    sun = _article("sun", SUN, paper="The Sun", state="New York")
    out, reason = _run("The Sun reported that all passengers were taken off safely.",
                       "ALL PASSEN-\nGERS TAKEN OFF SAFELY BY THE VIRGINIAN", sun)
    assert reason is None and out["match"] >= MATCH_THRESHOLD


def test_abbreviated_title_supports_full_word():
    art = _article("c", "Capt. E. J. Smith went down with his ship, according to survivors landed by the "
                        "Carpathia.", paper="The Salt Lake Herald", state="Utah")
    _, reason = _run("The Salt Lake Herald reported that Captain Smith went down with his ship.",
                     "Capt. E. J. Smith went down with his ship", art)
    assert reason is None


def test_paraphrased_hindsight_removed():
    _, reason = _run("Captain Smith went down with his ship.",
                     "Captain E. J. Smith, commander; H. T. Wilde, chief officer", _article("c", CREW))
    assert reason == "uses words its source does not contain: went, ship"


def test_future_tense_bankruptcy_removed():
    start = datetime(2008, 9, 10, tzinfo=UTC)
    cutoff = datetime(2008, 9, 12, 16, 10, tzinfo=UTC)
    text = ("Lehman Brothers was fighting for its survival last night as hopes grew that Bank of America "
            "would rescue the Wall Street bank.")
    art = _article("g", text, paper="The Guardian", state="", date="2008-09-12",
                   available_at="2008-09-12T15:40:00Z", page="")
    question = "Will Lehman Brothers be rescued?"
    allowed = allowed_terms(question, "4:10 p.m. BST, Friday, September 12, 2008", {"g": art}, start, cutoff)
    sentence = {"text": "Lehman will file for bankruptcy on Monday.",
                "quote": "hopes grew that Bank of America would rescue the Wall Street bank", "article_id": "g"}
    _, reason = check(sentence, text, art, cutoff, {"g"}, question, allowed)
    assert reason == "uses words its source does not contain: file, bankruptcy, monday"


def test_unsupported_for_labels_and_figures():
    assert unsupported("Sank with 1,500 lost", SALT_LAKE, QUESTION, set()) == \
        "states 1500, which its source does not mention"
    assert unsupported("All passengers safe", HERALD, QUESTION, set()) is None
    assert unsupported("Collision with the Carpathia", SALT_LAKE, QUESTION, {"carpathia"}) is None
    assert unsupported("Rescued by the Carpathia", SALT_LAKE, QUESTION, {"carpathia"}) == \
        "names Rescued, which its source does not mention"


# --- check(): display and contract fields ---------------------------------------------------------

def test_contract_fields_for_known_article():
    art = _article("e", STAR)
    out, reason = _run("The Evening Star reported that all passengers were taken off in the boats.",
                       "ALL PASSENGERS WERE TAKEN OFF in the boats", art)
    assert reason is None
    assert set(out) == {"text", "quote", "article_id", "paper", "date", "page", "url", "published", "place",
                        "match", "context", "hit"}
    assert out["article_id"] == "e" and out["paper"] == "The Evening Star" and out["date"] == "1912-04-15"
    assert out["published"] == "1912-04-15T23:59:59Z" and out["place"] == "District of Columbia"
    assert out["page"] == "1" and out["url"] == art["url"] and out["match"] >= MATCH_THRESHOLD


def test_hit_is_substring_of_context():
    filler = "The weather at Cape Race was clear and cold with a light wind from the west. " * 6
    art = _article("e", filler + STAR + " " + filler)
    out, _ = _run("The Evening Star reported that all passengers were taken off in the boats.",
                  "ALL PASSENGERS WERE TAKEN OFF in the boats", art, passage=STAR)
    assert out["hit"] == "ALL PASSEN-\nGERS WERE TAKEN OFF in the boats"
    assert out["hit"] in out["context"] and out["context"] in art["text"]
    assert len(out["context"]) <= CONTEXT_CHARS


def test_hit_in_context_keeps_raw_offsets_across_a_soft_hyphen():
    raw = "The fate of the steamer was un\u00adknown to the officers at Halifax. They sat in.the boat all night."
    art = _article("e", raw)
    out, reason = _run("The Evening Star reported that the steamer's fate was unknown to the officers.",
                       "The fate of the steamer was unknown to the officers", art)
    assert reason is None
    assert out["hit"] == "fate of the steamer was un\u00adknown to the officers"  # "the" is a light word
    assert out["hit"] in out["context"] and out["context"] in art["text"]
    assert quote_match("in the boat all night", raw) == 100


def test_hit_empty_and_context_is_quote_when_not_found():
    out, _ = _run("x", "the great liner sank at sea this morning", _article("s", SALT_LAKE))
    assert out["hit"] == "" and out["context"] == "the great liner sank at sea this morning"


def test_unknown_article_fields_are_contract_valid():
    sentence = {"text": "The Titanic is safe.", "quote": "the Titanic is safe and all aboard well",
                "article_id": "made-up"}
    out, reason = check(sentence, "", None, CUTOFF, {"e"}, QUESTION, set())
    assert reason == "article was not among the sources given"
    assert out == {"text": "The Titanic is safe.", "quote": "the Titanic is safe and all aboard well",
                   "article_id": "", "paper": "", "date": "1912-04-15", "page": "", "url": "",
                   "published": "1912-04-15T23:59:59+00:00",
                   "place": "", "match": 0, "context": "the Titanic is safe and all aboard well", "hit": ""}
    result = {"sentences": [out], "removed": [], "groups": [], "retrieved": []}
    assert validate_result(result, CUTOFF) == []


def test_check_never_raises_on_missing_fields():
    out, reason = check({}, "", None, CUTOFF, set(), "", set())
    assert reason == "quote shorter than 6 words" and out["text"] == "" and out["quote"] == ""
    sentence = {"text": None, "quote": None, "article_id": None}
    out, reason = check(sentence, None, {"article_id": "x"}, CUTOFF, {"x"}, None, set())
    assert reason == "quote shorter than 6 words"
    sentence = {"quote": "there is no cause for alarm, as the vessel", "article_id": "x"}
    out, reason = check(sentence, SALT_LAKE, {"article_id": "x", "text": SALT_LAKE}, CUTOFF, {"x"}, "", set())
    assert out["paper"] == "" and out["published"] == "" and isinstance(reason, (str, type(None)))


@pytest.mark.parametrize("bad", ["1912-04-15 junk", None, 12])
def test_bad_article_dates_count_as_unavailable(bad):
    art = {**_article("s", SALT_LAKE), "available_at": bad, "date": bad}
    _, reason = _run("The Salt Lake Herald said the vessel is unsinkable.",
                     "there is no cause for alarm, as the vessel is unsinkable", art)
    assert reason == "article was not published by the cutoff"


def test_match_threshold_follows_module_value(monkeypatch):
    monkeypatch.setattr(verify, "MATCH_THRESHOLD", 101)
    _, reason = _run("The Salt Lake Herald said the vessel is unsinkable.",
                     "there is no cause for alarm, as the vessel is unsinkable", _article("s", SALT_LAKE))
    assert reason == "quote not found in the article (best match 100%)"


# --- fix round 1: ellipsis stitching, negating prefixes and contractions, polarity ------------------

def test_ellipsis_stitch_of_scattered_words_fails():
    quote = "The Titanic struck an iceberg … All passengers … lost"
    assert quote_match(quote, STITCH) < MATCH_THRESHOLD
    sun = _article("sun", STITCH, paper="The Sun", state="New York")
    _, reason = _run("The Sun reported that the Titanic struck an iceberg and all passengers were lost.",
                     quote, sun)
    assert reason is not None and reason.startswith("quote not found in the article")


def test_ellipsis_part_limits():
    two = "The Titanic struck an iceberg near the Banks … the officers say the liner lost no time"
    assert quote_match(two, STITCH) >= MATCH_THRESHOLD
    few_significant = "The Titanic struck an iceberg near the Banks … is in a berg"
    assert quote_match(few_significant, SALT_LAKE + " " + STITCH) < MATCH_THRESHOLD
    filler = " ".join(f"word{i}" for i in range(45)) + "."
    near = "The Titanic struck an iceberg near the Banks. " + " ".join(f"word{i}" for i in range(30)) + \
        ". All passengers are reported safe tonight."
    far = "The Titanic struck an iceberg near the Banks. " + filler + " All passengers are reported safe tonight."
    gap = "The Titanic struck an iceberg … All passengers are reported safe"
    assert quote_match(gap, near) >= MATCH_THRESHOLD
    assert quote_match(gap, far) < MATCH_THRESHOLD
    four = ("The Titanic struck an iceberg … All passengers are reported safe … the officers say the liner "
            "… sending wireless calls for aid")
    assert quote_match(four, STITCH) < MATCH_THRESHOLD
    three = ("The Titanic struck an iceberg … All passengers are reported safe … sending wireless calls "
             "for aid")
    assert quote_match(three, STITCH) >= MATCH_THRESHOLD


@pytest.mark.parametrize("passage, quote", [
    ("Wireless messages say all the passengers are safe and well tonight.",
     "messages say all the passengers are unsafe and well"),
    (MISSOULIAN, "The steamer is believed to be hurt and her people"),
    (SALT_LAKE, "there is no cause for alarm, as the vessel is sinkable"),
    ("The fate of the liner is known to the officers of the line.",
     "The fate of the liner is unknown to the officers"),
    ("The officers could say where the liner was at midnight.",
     "The officers couldn't say where the liner was at midnight"),
    ("The officers could say where the liner was at midnight.",
     "The officers couldnt say where the liner was at midnight"),
    ("The boats were lowered in good time, the officers say.", "The boats weren't lowered in good time"),
])
def test_negating_prefixes_and_contractions_fail(passage, quote):
    assert quote_match(quote, passage) < MATCH_THRESHOLD


def test_contractions_match_their_expansion():
    assert quote_match("The officers couldn't say where the liner was", "The officers could not say where the "
                       "liner was at midnight.") >= MATCH_THRESHOLD
    assert quote_match("The officers could not say where the liner was", "The officers couldnt say where the "
                       "liner was at midnight.") >= MATCH_THRESHOLD
    assert quote_match("are on the way to New York aboard the Carpathia", STAR) >= MATCH_THRESHOLD
    assert quote_match("ALL PASSENGERS WERE TAKEN OFF in the boats", STAR) >= MATCH_THRESHOLD


HERALD_QUOTE = "wireless messages say that all pas- sengers are safe"


@pytest.mark.parametrize("question, text, reason", [
    ("Did the Titanic sink?", "The Sun reported that the Titanic did sink.",
     "uses words its source does not contain: sink"),
    ("Did the Titanic sink?", "The Sun reported that the Titanic did not sink.", "denies what its source says"),
    (QUESTION, "The Sun reported that the passengers were not saved.", "denies what its source says"),
    (QUESTION, "No passengers were saved, the Sun reported.", "denies what its source says"),
    ("Did anyone survive the Titanic?", "The Sun reported that no one survived the Titanic.",
     "denies what its source says"),
    (QUESTION, "The Sun reported that the passengers weren't safe.", "denies what its source says"),
])
def test_polarity_and_question_words_removed(question, text, reason):
    sun = _article("sun", HERALD, paper="The Sun", state="New York")
    _, got = _run(text, HERALD_QUOTE, sun, question=question)
    assert got == reason


def test_dropped_negation_removed():
    art = _article("m", MISSOULIAN, paper="The Daily Missoulian", state="Montana")
    _, reason = _run("The Daily Missoulian said her people are in danger.",
                     "The steamer is believed to be unhurt and her people are in no danger", art)
    assert reason == "drops a negation its source contains"


def test_honest_negative_mirroring_its_quote_kept():
    art = _article("m", MISSOULIAN, paper="The Daily Missoulian", state="Montana")
    _, reason = _run("The Daily Missoulian said her people are in no danger.",
                     "The steamer is believed to be unhurt and her people are in no danger", art)
    assert reason is None


@pytest.mark.parametrize("text", [
    "No paper yet reports whether the Titanic sank.",
    "The papers do not yet report whether the Titanic sank.",
    "The Sun did not report whether the Titanic sank.",
    "Nothing in the papers yet says that the Titanic sank.",
    "The papers have not yet reported whether the Titanic sank.",
    "It is not yet known whether the Titanic sank.",
    "No word yet whether the Titanic sank.",
])
def test_absence_statements_kept(text):
    sun = _article("sun", HERALD, paper="The Sun", state="New York")
    _, reason = _run(text, HERALD_QUOTE, sun, question="Did the Titanic sink?")
    assert reason is None


def test_not_yet_stating_the_outcome_removed():
    sun = _article("sun", HERALD, paper="The Sun", state="New York")
    _, reason = _run("The Titanic has not yet sunk, the Sun reported.", HERALD_QUOTE, sun,
                     question="Did the Titanic sink?")
    assert reason == "denies what its source says"


def test_question_words_before_an_absence_frame_removed():
    sun = _article("sun", HERALD, paper="The Sun", state="New York")
    _, reason = _run("The Titanic sank, but no paper reports it yet.", HERALD_QUOTE, sun,
                     question="Did the Titanic sink?")
    assert reason == "uses words its source does not contain: sank"


def test_unsupported_polarity_needs_quote_only_for_dropped_negation():
    assert unsupported("Her people are in danger", MISSOULIAN, QUESTION, set()) is None
    assert unsupported("Her people are in danger", MISSOULIAN, QUESTION, set(),
                       quote="her people are in no danger") == "drops a negation its source contains"
    assert unsupported("The passengers are not safe", HERALD, QUESTION, set()) == "denies what its source says"


# --- fix round 2: capitalised question words, clause scope of absence frames, minors -----------------

@pytest.mark.parametrize("question, text, word", [
    ("Were passengers drowned?", "Drowned passengers are safe, the Sun reported.", "drowned"),
    ("Were there survivors?", "Survivors are safe, the Sun reported.", "survivors"),
    (QUESTION, "Saved were all the passengers, the Sun reported.", "saved"),
])
def test_capitalised_question_words_are_not_names(question, text, word):
    sun = _article("sun", HERALD, paper="The Sun", state="New York")
    _, reason = _run(text, HERALD_QUOTE, sun, question=question)
    assert reason == f"uses words its source does not contain: {word}"


@pytest.mark.parametrize("text", [
    "The Sun did not report it, but the Titanic sank.",
    "No paper yet reports it, but the Titanic sank.",
    "The Sun has not yet reported it, but the Titanic sank.",
    "Nothing yet, but the Titanic sank.",
    "No papers say so, yet the Titanic sank.",
    "No paper yet reports whether the Titanic sank; it sank.",
    "No paper yet reports that the Titanic sank, but it did.",
    "No paper yet reports whether the Titanic sank — it sank.",
    "No paper yet reports whether the Titanic sank. It sank.",
])
def test_absence_frame_ends_at_the_clause(text):
    sun = _article("sun", HERALD, paper="The Sun", state="New York")
    _, reason = _run(text, HERALD_QUOTE, sun, question="Did the Titanic sink?")
    assert reason is not None


def test_absence_frame_survives_abbreviation_periods():
    sun = _article("sun", HERALD, paper="The Sun", state="New York")
    _, reason = _run("No paper yet reports whether Capt. Smith's Titanic sank.", HERALD_QUOTE, sun,
                     question="Did Captain Smith's Titanic sink?")
    assert reason is None


def test_without_phrase_may_be_omitted():
    text = "The liner proceeded without assistance to Halifax, where she arrived this morning."
    art = _article("h", text, paper="The Sun", state="New York")
    _, reason = _run("The Sun said the liner proceeded to Halifax.",
                     "The liner proceeded without assistance to Halifax", art)
    assert reason is None


def test_mis_prefix_is_not_ocr_damage():
    assert not verify._same("misreported", "reported")
    assert quote_match("the loss was reported by the line tonight",
                       "the loss was misreported by the line tonight.") < MATCH_THRESHOLD


# --- fix round 3: question-only words only in a whether/if/that complement; names by sentence position --

@pytest.mark.parametrize("text", [
    "No paper yet reports it, and the Titanic sank.",
    "No paper yet reports it and the Titanic sank.",
    "No paper yet reports whether the Titanic sank, it sank.",
    '"No word yet," but the Titanic sank.',
    "“No word yet,” but the Titanic sank.",
    "No papers say so, and yet the Titanic sank.",
    "No papers say so yet the Titanic sank.",
    "No paper yet reports it - the Titanic sank.",
    "No paper yet reports it – the Titanic sank.",
    "No paper yet reports it — the Titanic sank.",
    "No paper yet reports it (the Titanic sank).",
    "No paper yet reports it, so the Titanic sank.",
    "No word yet, as the Titanic sank.",
    "No paper yet reports it because the Titanic sank.",
    "No paper yet reports it, the Titanic sank.",
    "No paper yet reports it.The Titanic sank.",
    "No paper yet reports whether the Titanic sank or not.",
    "No paper yet reports whether the Titanic sank (it sank).",
    "Nothing in the papers yet says the Titanic sank.",
])
def test_question_words_need_a_closed_complement(text):
    sun = _article("sun", HERALD, paper="The Sun", state="New York")
    _, reason = _run(text, HERALD_QUOTE, sun, question="Did the Titanic sink?")
    assert reason is not None


@pytest.mark.parametrize("question, text", [
    ("Did the Titanic sink?", "No paper yet reports whether the Titanic sank."),
    ("Did Captain Smith's Titanic sink?", "No paper yet reports whether Capt. Smith's Titanic sank."),
    ("Did the Titanic reach St. John's?", "No paper yet reports whether the Titanic reached St. John's."),
])
def test_closed_complement_after_absence_frame_kept(question, text):
    sun = _article("sun", HERALD, paper="The Sun", state="New York")
    _, reason = _run(text, HERALD_QUOTE, sun, question=question)
    assert reason is None


LINER = "The White Star liner struck an iceberg on her maiden voyage, wireless messages say."


@pytest.mark.parametrize("question", ["did the titanic sink?", "Titanic: did she sink?"])
def test_question_names_allowed_mid_sentence_whatever_the_question_case(question):
    sun = _article("sun", LINER, paper="The Sun", state="New York")
    _, reason = _run("The Sun reported that the Titanic struck an iceberg.",
                     "The White Star liner struck an iceberg", sun, question=question)
    assert reason is None


def test_question_name_at_sentence_start_needs_source_or_question_capitals():
    text = "Captain Smith of the liner is safe and on the bridge, wireless messages say."
    sun = _article("sun", text, paper="The Sun", state="New York")
    _, reason = _run("Smith is safe, the Sun reported.", "Captain Smith of the liner is safe", sun,
                     question="is captain smith safe?")
    assert reason is None
    titanic = _article("t", LINER, paper="The Sun", state="New York")
    _, reason = _run("Titanic struck an iceberg, the Sun reported.", "The White Star liner struck an iceberg",
                     titanic, question="Did the Titanic sink?")
    assert reason is None  # the question itself capitalises "Titanic" after its first word


def test_title_before_a_question_name_allowed():
    text = "Smith of the liner is safe and on the bridge, wireless messages from the ship say."
    sun = _article("sun", text, paper="The Sun", state="New York")
    _, reason = _run("Captain Smith is safe, the Sun reported.", "Smith of the liner is safe and on the bridge",
                     sun, question="Is Captain Smith safe?")
    assert reason is None


# --- fix round 4: no denial verbs in the frame window; question casing decides names; factive "known that" --

DENIAL = ("The White Star line denied reports that the liner was in danger, and its officers doubted the "
          "dispatches; the line doubts them still, wireless messages say.")
DENIAL_QUOTE = "denied reports that the liner was in danger"


@pytest.mark.parametrize("text", [
    "No paper has yet denied reports that the Titanic sank.",
    "No paper yet doubts reports that the Titanic sank.",
    "No dispatch has yet doubted reports that the Titanic sank.",
    "No paper has yet denied that the Titanic sank.",
    "Nothing in the papers yet doubts that the Titanic sank.",
])
def test_denial_verb_in_the_frame_window_removed(text):
    sun = _article("sun", DENIAL, paper="The Sun", state="New York")
    _, reason = _run(text, DENIAL_QUOTE, sun, question="Did the Titanic sink?")
    assert reason == "uses words its source does not contain: sank"


def test_do_not_frame_needs_a_reporting_verb():
    sun = _article("sun", DENIAL, paper="The Sun", state="New York")
    _, reason = _run("The papers do not yet doubt reports that the Titanic sank.", DENIAL_QUOTE, sun,
                     question="Did the Titanic sink?")
    assert reason == "denies what its source says"


@pytest.mark.parametrize("text", [
    "No paper has yet reported whether the Titanic sank.",
    "No report yet says whether the Titanic sank.",
    "No reports yet say whether the Titanic sank.",
    "Nothing in the papers yet says whether the Titanic sank.",
    "Nothing in any paper yet reports whether the Titanic sank.",
    "It has not yet been reported whether the Titanic sank.",
    "The Sun does not yet say whether the Titanic sank.",
])
def test_frame_words_before_the_reporting_word_kept(text):
    sun = _article("sun", HERALD, paper="The Sun", state="New York")
    _, reason = _run(text, HERALD_QUOTE, sun, question="Did the Titanic sink?")
    assert reason is None


SMITH = "Smith of the liner is safe and on the bridge, wireless messages from the ship say."


@pytest.mark.parametrize("question, text, reason", [
    ("Is Captain Smith dead?", "The Sun reported that Captain Smith is Dead.",
     "uses words its source does not contain: dead"),
    ("Is Captain Smith dead?", "Captain Smith is Dead, the Sun reported.",
     "uses words its source does not contain: dead"),
    ("Were passengers drowned?", "The Sun reported all Drowned.",
     "uses words its source does not contain: drowned"),
    ("did many drown?", "The Sun reported that Many did Drown.",
     "uses words its source does not contain: drown"),
    ("Did the Titanic sink?", "The Sun reported that the Titanic Sank.",
     "names Sank, which its source does not mention"),
])
def test_question_casing_decides_names(question, text, reason):
    sun = _article("sun", SMITH, paper="The Sun", state="New York")
    _, got = _run(text, "Smith of the liner is safe and on the bridge", sun, question=question)
    assert got == reason


def test_question_without_casing_uses_the_position_rule():
    sun = _article("sun", SMITH, paper="The Sun", state="New York")
    _, reason = _run("Captain Smith is safe, the Sun reported.", "Smith of the liner is safe and on the bridge",
                     sun, question="Is Captain Smith safe?")
    assert reason is None
    liner = _article("t", LINER, paper="The Sun", state="New York")
    for question in ("did the titanic sink?", "Titanic: did she sink?"):
        _, reason = _run("The Sun reported that the Titanic struck an iceberg.",
                         "The White Star liner struck an iceberg", liner, question=question)
        assert reason is None


@pytest.mark.parametrize("text", [
    "It is not yet known that the Titanic sank.",
    "It is not yet clear that the Titanic sank.",
    "It is not yet certain that the Titanic sank.",
    "It is not yet evident that the Titanic sank.",
])
def test_factive_that_complement_removed(text):
    sun = _article("sun", HERALD, paper="The Sun", state="New York")
    _, reason = _run(text, HERALD_QUOTE, sun, question="Did the Titanic sink?")
    assert reason == "uses words its source does not contain: sank"


@pytest.mark.parametrize("text", [
    "It is not yet known whether the Titanic sank.",
    "It is not yet known if the Titanic sank.",
    "It is not yet certain whether the Titanic sank.",
    "It is not yet clear if the Titanic sank.",
    "No paper yet reports that the Titanic sank.",
])
def test_factive_whether_complement_kept(text):
    sun = _article("sun", HERALD, paper="The Sun", state="New York")
    _, reason = _run(text, HERALD_QUOTE, sun, question="Did the Titanic sink?")
    assert reason is None


@pytest.mark.parametrize("text", [
    "It is not yet known that the big ship is being towed to Halifax.",
    "It is not yet certain that all passengers are safe.",
    "It has not yet been confirmed that the big ship is being towed to Halifax.",
])
def test_factive_that_with_source_backed_complement_kept(text):
    sun = _article("sun", HERALD, paper="The Sun", state="New York")
    _, reason = _run(text, HERALD_QUOTE, sun, question="Did the Titanic sink?")
    assert reason is None


# --- fix round 5: a mixed-case question names only the words it capitalises after its first word ---------

@pytest.mark.parametrize("question, text, source, bad", [
    ("Were passengers drowned?", "The Sun reported that the passengers Drowned.", HERALD, "drowned"),
    ("Were passengers drowned?", "The Sun reported that Passengers Drowned.", SMITH, "passengers, drowned"),
    ("Is the captain dead?", "The Sun reported the captain Dead.", SMITH, "captain, dead"),
    ("Is the captain dead?", "The Sun said the captain of the Titanic Dead.", HERALD, "captain, dead"),
    ("Were the passengers lost?", "The Sun reported the passengers Lost.", HERALD, "lost"),
    ("Did the passengers die?", "The Sun reported that the passengers Die.", HERALD, "die"),
    ("Did the ship sink?", "The Sun reported the Titanic Sink.", HERALD, "sink"),
    ("IS SMITH DEAD?", "Smith is Dead, the Sun reported.", SMITH, "dead"),
    ("IS SMITH DEAD?", "The Sun reported that Smith is Dead.", SMITH, "dead"),
])
def test_sentence_case_question_words_are_not_names(question, text, source, bad):
    reason = unsupported(text, source, question, {"sun"}, quote="x")
    assert reason == f"uses words its source does not contain: {bad}"


@pytest.mark.parametrize("question, text, source", [
    ("did the titanic sink?", "The Sun reported that the Titanic struck an iceberg.", LINER),
    ("Titanic: did she sink?", "The Sun reported that the Titanic struck an iceberg.", LINER),
    ("Is Captain Smith safe?", "Captain Smith is safe, the Sun reported.", SMITH),
    ("was captain smith on the bridge?", "The Sun reported that Captain Smith was on the bridge.", SMITH),
    ("WAS SMITH ON THE BRIDGE?", "The Sun reported that Smith was on the bridge.", SMITH),
])
def test_question_names_kept_under_the_casing_rule(question, text, source):
    assert unsupported(text, source, question, {"sun"}, quote="x") is None


# --- tuning on the real-data check (each case is from the November 5, 2026 run) -------------------

def test_web_paper_name_allows_its_plain_name():
    text = "the latest official update on the search operation was several hours ago."
    art = _article("g", text, paper="theguardian.com")
    _, reason = _run("The latest official update on the search operation was several hours ago, according "
                     "to The Guardian.", "The latest official update on the search operation was several hours ago",
                     art, question="What is the latest news from the search?")
    assert reason is None


def test_connectives_and_hedges_need_no_source():
    text = ("the white star line officials today issued an official statement in which they stated that the "
            "great liner was unsinkable. With its 1900 passengers safely transferred to another vessel, the "
            "Titanic would be towed to port safely.")
    art = _article("w", text, paper="The Washington times")
    for sentence in ("The Washington Times likewise reported that the great liner was unsinkable.",
                     "The paper reported that the Titanic's roughly 1900 passengers were safely transferred.",
                     "The paper reported the Titanic would be towed toward port safely.",
                     "The paper relayed a statement that the great liner was unsinkable."):
        _, reason = _run(sentence, "officials today issued an official statement in which they stated", art)
        assert reason is None, sentence


def test_ocr_v_for_w_in_quotes():
    passage = "Fifteen vater-tight bulkheads divide the great vessel. making her unsinkable"
    assert quote_match("Fifteen water-tight bulkheads divide the great vessel", passage) >= MATCH_THRESHOLD
    # a sentence word is never supported by a joined pair under OCR confusions, so "watertight" needs the
    # source's own spelling
    _, reason = _run("The paper reported that fifteen watertight bulkheads divide the vessel.", passage,
                     _article("r", passage))
    assert reason == "uses words its source does not contain: watertight"


@pytest.mark.parametrize("passage, quote", [
    # "tho" for "the"
    ("tho latest reports indicate the transference of passengers is beng carried on successfully",
     "The latest reports indicate The transference of passengers IS beng carried on successfully"),
    # "the" glued to the word before it
    ("new york, april l5athe white star line officials, after a conference lasting more than an hour",
     "The White Star Line officials, after a conference lasting more than an hour"),
    # "the" misread as "me" or "tn"
    ("it is certain that the titanic went down. me survivors aboard ffe carparhia we're picked up from iifeboats",
     "the survivors aboard ffe carparhia we're picked up from iifeboats"),
    ("capt campbell reported that tn virginian's rescue trip was fruitless the message is",
     "the Virginian's rescue trip was fruitless the message is"),
    # a word split by OCR with a letter misread
    ("canso. n. s. april 15.-the tt. tanic. having transferred all her passengers at 2 o'clock",
     "The Titanic having transferred all her passengers at 2 O'Clock"),
])
def test_ocr_garbled_the_and_split_words_match(passage, quote):
    assert quote_match(quote, passage) >= MATCH_THRESHOLD


def test_the_and_split_words_still_need_every_other_word():
    passage = "tho latest reports indicate the transference of passengers is beng carried on successfully"
    assert quote_match("The latest reports indicate no transference of passengers is beng carried", passage) \
        < MATCH_THRESHOLD
    assert quote_match("all the latest reports indicate the transference of passengers", passage) < MATCH_THRESHOLD
    # a whole neighbouring word is still not absorbed by a join
    assert quote_match("the passengers were all reported safe and well", "the allpassengers were reported safe "
                       "and well") < MATCH_THRESHOLD


def test_irregular_past_forms_support_each_other():
    text = ("Lehman Brothers was last night scrambling to secure a buyer after a week in which confidence in "
            "the bank has fallen at a breathtaking pace.")
    _, reason = _run("Lehman Brothers was scrambling to secure a buyer as confidence in the bank fell at a "
                     "breathtaking pace.", "Lehman Brothers was last night scrambling to secure a buyer",
                     _article("l", text, paper="The Guardian"), question="Will Lehman survive the weekend?")
    assert reason is None
    sank = "More than 1900 persons. it is believed, sank to death early Tuesday"
    _, reason = _run("The paper reported that more than 1900 persons were believed to have sunk to death.",
                     sank, _article("h", sank), question="How many people died on the Titanic?")
    assert reason is None
    found = "the steamer olympic reports that carpathia reached position at daybreak and found wreckage only"
    _, reason = _run("The paper reported the Carpathia finding wreckage only.", found, _article("o", found))
    assert reason is None


def test_present_form_in_source_does_not_support_past_form():
    sinking = "Half an hour afterwards another message came reporting that the Titanic is sinking by the head."
    _, reason = _run("The paper reported that the Titanic sank by the head.",
                     "another message came reporting that the Titanic is sinking by the head", _article("s", sinking),
                     question="What has happened to the Titanic?")
    assert reason == "uses words its source does not contain: sank"
    makes = "In a partial list of survivors is the name of Mrs. Astor. This makes it almost certain."
    _, reason = _run("The paper said this made it almost certain.", "In a partial list of survivors is the name",
                     _article("m", makes), question="Is John Jacob Astor safe?")
    assert reason == "uses words its source does not contain: made"


def test_takeover_supported_by_taken_over():
    text = ("Lehman Brothers yesterday saw its shares fall 40% in early trading. The market is guessing that "
            "Lehman will be taken over at a knock-down price, probably within days.")
    _, reason = _run("The paper said the market guessed a takeover at a knock-down price.",
                     "The market is guessing that Lehman will be taken over at a knock-down price",
                     _article("t", text, paper="The Guardian"), question="How far have Lehman's shares fallen?")
    assert reason is None


def test_unknown_article_at_a_minute_cutoff_is_contract_valid():
    cutoff = datetime(2023, 6, 21, 18, 0, tzinfo=UTC)
    sentence = {"text": "Hope remains.", "quote": "So there's definitely hope that the crew is alive",
                "article_id": "guardian:mangled"}
    out, reason = check(sentence, "", None, cutoff, {"g"}, "Will they be found in time?", set())
    assert reason == "article was not among the sources given"
    result = {"sentences": [out], "removed": [], "groups": [], "retrieved": []}
    assert validate_result(result, cutoff) == []


def test_ocr_punctuation_does_not_lower_the_quote_score():
    passage = ("april l9 the total =: toll of lives lost in ihe titanic , g disaster number sixteen hundred : y "
               "and one tie steamer")
    quote = "The total toll Of lives lost in the Titanic disaster number sixteen hundred and one."
    assert quote_match(quote, passage) >= MATCH_THRESHOLD
    assert quote_match(quote.replace("sixteen", "seventeen"), passage) < MATCH_THRESHOLD


def test_number_word_split_by_ocr_is_joined():
    text = "new york, april l6-eight hun dred and SiXty-siX survivors Of the liner Titanic"
    assert "866" in numbers(text)
    _, reason = _run("The paper reported that eight hundred and sixty-six survivors of the liner Titanic are "
                     "on the Carpathia.", "eight hun dred and SiXty-siX survivors Of the liner Titanic",
                     _article("l", text + " are now on the steamer Carpathia"))
    assert reason is None
    assert numbers("one hundred") == {"100"} and numbers("two thou sand") == {"2000"}


@pytest.mark.parametrize("text", ["it's even worse", "the ship's even keel", "there's even hope",
                                  "on e boat", "el even men", "for ty miles", "six teen"])
def test_split_words_do_not_invent_numbers(text):
    assert numbers(text) & {"7", "1", "11", "40", "16"} == set()
    assert numbers("it's even") == set()


def test_night_and_day_supported_by_tonight_and_today():
    text = ("Virginian and Partisan had made no report to the company's headquarters here up 10 10 o'clock "
            "tonight of what they had done")
    _, reason = _run("The paper reported that the Virginian and Partisan had made no report to the company's "
                     "headquarters by 10 o'clock that night.",
                     "had made no report TO the company's headquarters here up 10 10 o'clock tonight",
                     _article("w", text))
    assert reason is None


def test_ocr_damaged_name_in_source_supports_the_name():
    text = ("the steamship offices received a wireless from the carpaihla this morning saying that mrs john "
            "jacob astor and her maid we're among the survivors aboard.")
    _, reason = _run("The paper reported that a wireless from the Carpathia said Mrs John Jacob Astor and her "
                     "maid were among the survivors aboard.",
                     "saying that mrs john jacob astor and her maid we're among the survivors aboard",
                     _article("c", text), question="Is John Jacob Astor safe?")
    assert reason is None



@pytest.mark.parametrize("passage, quote", [
    ("the steamer carried an armed guard on deck", "the steamer carried unarmed guard on deck"),
    ("the wireless was an active station that night", "the wireless was inactive station that night"),
    ("the transference of passengers is beng carried on successfully", "the transference of passengers is beng "
     "carried unsuccessfully"),
])
def test_join_never_absorbs_a_negating_prefix(passage, quote):
    assert quote_match(quote, passage) < MATCH_THRESHOLD


@pytest.mark.parametrize("name, word", [("Hill", "bill"), ("Ball", "hall"), ("Hunt", "bunt"), ("Butt", "hutt")])
def test_short_names_not_matched_by_ocr_skeleton(name, word):
    text = f"the {word} was paid at the office yesterday"
    _, reason = _run(f"The paper reported that {name} was paid at the office.", text, _article("n", text))
    assert reason == f"names {name}, which its source does not mention"


@pytest.mark.parametrize("word, source_word", [
    ("displeased", "is pleased"), ("unarmed", "an armed"), ("found", "bound"), ("sound", "round"),
    ("saved", "waved"), ("lives", "wives"), ("killed", "filled"), ("drowned", "crowned"), ("alive", "olive"),
    ("missing", "kissing"),
])
def test_ocr_confusions_do_not_support_other_words(word, source_word):
    source = f"The despatch from the steamer said the passengers {source_word} and the steamer went on."
    reason = unsupported(f"The despatch said the passengers {word}.", source, "Any news of the steamer?", set())
    assert reason is not None and word in reason, reason


@pytest.mark.parametrize("text, source", [
    # Q03 at April 16, 1912: "picked up from iifeboats"
    ("The paper reported that the survivors were picked up from lifeboats.",
     "the survivors aboard ffe carparhia we're picked up from iifeboats"),
    # Q02 at April 15, 1912: "the titanic is still afioat"
    ("The paper reported that the Titanic is still afloat.",
     "while badly damaged the titanic is still afioat and is making its way toward halifax"),
    # Q04 at April 16, 1912: "promlnent american milllonaires"
    ("The paper reported that other prominent American millionaires went down with the vessel.",
     "col. john jacob astor and other promlnent american milllonaires prob. abs went down with the vessel."),
])
def test_ocr_confusions_support_the_corrected_word(text, source):
    assert unsupported(text, source, "What has happened to the Titanic?", set()) is None


@pytest.mark.parametrize("passage, quote, sentence, word", [
    ("the olympic was in active service on the atlantic route",
     "the olympic was inactive service on the atlantic route", "The paper reported that the Olympic was inactive.",
     "inactive"),
    ("the crew were in capable hands of the captain", "the crew were incapable hands of the captain",
     "The paper reported that the crew were incapable.", "incapable"),
    ("the man was in sane condition when found", "the man was insane condition when found",
     "The paper reported that the man was insane.", "insane"),
    ("it was a typical day at the white star offices", "it was atypical day at the white star offices",
     "The paper reported that it was an atypical day.", "atypical"),
])
def test_standalone_prefix_word_never_joins_into_a_negation(passage, quote, sentence, word):
    assert quote_match(quote, passage) < MATCH_THRESHOLD
    reason = unsupported(sentence, passage, "What did the papers say?", set())
    assert reason is not None and word in reason, reason


@pytest.mark.parametrize("passage, quote, sentence", [
    ("the passengers were un harmed by the collision", "the passengers were unharmed by the collision",
     "The paper reported that the passengers were unharmed."),
    ("the steamer was at a dis tance from the wreck", "the steamer was at a distance from the wreck",
     "The paper reported that the steamer was at a distance from the wreck."),
])
def test_real_ocr_splits_still_join(passage, quote, sentence):
    assert quote_match(quote, passage) >= MATCH_THRESHOLD
    assert unsupported(sentence, passage, "What did the papers say?", set()) is None


_REAL_WORD_PAIRS = [("hanged", "banged"), ("hidden", "bidden"), ("firing", "tiring"), ("fearing", "tearing"),
                    ("hearing", "bearing"), ("toiled", "foiled"), ("tolled", "toiled"), ("tilled", "filled"),
                    ("tasted", "fasted"), ("hailed", "bailed"), ("tender", "fender"), ("heating", "beating"),
                    ("falling", "failing")]


# Both directions; "fearing" itself needs no source (an attribution word, like "feared"), so only
# "tearing" from "fearing" is checked for that pair.
@pytest.mark.parametrize("word, source_word", [p for a, b in _REAL_WORD_PAIRS for p in ((a, b), (b, a))
                                               if p[0] != "fearing"])
def test_real_words_do_not_support_each_other_by_ocr_confusion(word, source_word):
    source = f"The despatch from the steamer said the passengers were {source_word} and the steamer went on."
    reason = unsupported(f"The despatch said the passengers were {word}.", source, "Any news of the steamer?", set())
    assert reason is not None and word in reason, reason


@pytest.mark.parametrize("passage, quote, sentence, word", [
    ("the captain was not able to reach the bridge in time", "the captain was notable to reach the bridge in time",
     "The paper reported that the notable captain reached the bridge.", "notable"),
    ("the lookout did not ice the deck before the watch began", "the lookout did notice the deck before the watch began",
     "The paper reported that the lookout did notice the deck.", "notice"),
])
def test_negation_word_never_joins_away(passage, quote, sentence, word):
    assert quote_match(quote, passage) < MATCH_THRESHOLD
    reason = unsupported(sentence, passage, "What did the papers say?", set())
    assert reason is not None and word in reason, reason


def test_negation_word_never_joins_in():
    passage = "the notable captain reached the bridge in good time"
    assert quote_match("the not able captain reached the bridge in good time", passage) < MATCH_THRESHOLD
    assert unsupported("The paper reported that the captain was not notable.", passage,
                       "What did the papers say?", set()) is not None


def test_can_not_still_joins_into_cannot():
    passage = "the officers said they can not give a list of the survivors"
    assert quote_match("the officers said they cannot give a list of the survivors", passage) >= MATCH_THRESHOLD
    assert unsupported("The paper reported that the officers cannot give a list of the survivors.", passage,
                       "What did the papers say?", set()) is None


@pytest.mark.parametrize("sentence, source", [
    ("The paper reported that the Inactive Olympic was in port.", "the active olympic was in port that night"),
    ("The paper reported that the Incapable captain was on the bridge.",
     "the capable captain was on the bridge all night"),
])
def test_ocr_name_match_keeps_the_negating_prefix_guard(sentence, source):
    assert unsupported(sentence, source, "What did the papers say?", set()) is not None
