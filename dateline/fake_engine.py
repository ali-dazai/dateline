"""Fake engine: canned answers that follow the engine contract (v2, with events).

Lets the web app and the evaluation pipeline run before the real engine is available.
Switch to the real engine by setting DATELINE_ENGINE=real in .env.

1912: quotes are real April 1912 headlines (Molly Brown House Museum, digitalnc.org);
article IDs and links are placeholders. 2008 and 2023: quotes, IDs, times and URLs are
real items from data/articles_<event>.parquet (Guardian Open Platform). Not for the experiment.
"""
from __future__ import annotations

from datetime import date, datetime

from .contract import AnswerResult
from .events import DEFAULT_EVENT, cutoff_label, parse_cutoff

_LOC = "https://www.loc.gov/collections/chronicling-america/"
_G = "https://www.theguardian.com/"
_NO_QUOTE = "No matching quote in any article dated on or before the cutoff"


def _s(text, quote, article_id, paper, published, page, url, claim):
    return {"text": text, "quote": quote, "article_id": article_id, "paper": paper,
            "date": published[:10], "published": published, "page": page, "url": url,
            "_claim": claim}


# Each story: (sentences with the claim they belong to, the hindsight slip, when it stops being hindsight)
STORIES: dict[str, dict] = {
    "titanic-1912": {
        "sentences": [
            _s("Several papers report that everyone aboard the Titanic was saved after the collision.",
               "ALL SAVED FROM TITANIC AFTER COLLISION", "FAKE-0001", "The Evening Sun (New York)",
               "1912-04-15", "1", _LOC, "All passengers saved; ship towed to port"),
            _s("The Christian Science Monitor says the passengers were moved safely and the ship taken in tow.",
               "PASSENGERS SAFELY MOVED AND STEAMER TITANIC TAKEN IN TOW", "FAKE-0002",
               "The Christian Science Monitor", "1912-04-15", "1", _LOC,
               "All passengers saved; ship towed to port"),
            _s("Later reports say more than 1,500 people are feared to have died.",
               "More than 1,500 persons, it is feared, sank to death", "FAKE-0003",
               "The Carolina Union Farmer", "1912-04-18", "11",
               "https://newspapers.digitalnc.org/lccn/sn93065726/1912-04-18/ed-1/seq-11/",
               "About 1,500 feared lost"),
        ],
        "hindsight": "In the end, about 1,500 people died when the ship sank.",
        "known_from": "1912-04-18",
    },
    "lehman-2008": {
        "sentences": [
            _s("On Friday, markets rallied on hopes that Bank of America would rescue Lehman.",
               "growing hopes that Bank of America would rescue Lehman revived the Dow Jones Industrial Average",
               "guardian:business/marketforceslive/2008/sep/12/wolseley.jsainsbury", "The Guardian",
               "2008-09-12T16:10:02Z", "", _G + "business/marketforceslive/2008/sep/12/wolseley.jsainsbury",
               "A rescue buyer is expected"),
            _s("The Guardian reported that Lehman was scrambling to secure a buyer.",
               "Lehman Brothers was tonight scrambling to secure a buyer",
               "guardian:business/2008/sep/12/lehmanbrothers.creditcrunch", "The Guardian",
               "2008-09-12T17:32:31Z", "", _G + "business/2008/sep/12/lehmanbrothers.creditcrunch",
               "A rescue buyer is expected"),
            _s("By Monday morning, Lehman Brothers had filed for bankruptcy.",
               "Lehman Brothers finally threw in its towel and filed for bankruptcy",
               "guardian:commentisfree/cifamerica/2008/sep/15/useconomy.creditcrunch", "The Guardian",
               "2008-09-15T06:36:38Z", "", _G + "commentisfree/cifamerica/2008/sep/15/useconomy.creditcrunch",
               "Lehman files for bankruptcy"),
        ],
        "hindsight": "No buyer was found and Lehman filed for bankruptcy on 15 September.",
        "known_from": "2008-09-15T06:36:38Z",
    },
    "titan-2023": {
        "sentences": [
            _s("US media report that search crews heard banging sounds every 30 minutes.",
               "Search crews have heard banging sounds at 30-minute intervals, according to US media.",
               "guardian:uk-news/live/2023/jun/20/titanic-submarine-rescuers-missing-titan-submersible-search-latest-news#64927a498f08dd774dcd9fe2",
               "The Guardian", "2023-06-21T05:39:14Z", "",
               _G + "uk-news/live/2023/jun/20/titanic-submarine-rescuers-missing-titan-submersible-search-latest-news"
               "?page=with:block-64927a498f08dd774dcd9fe2#block-64927a498f08dd774dcd9fe2",
               "Noises suggest the crew may be alive"),
            _s("The US Coast Guard says a remotely operated vehicle found a debris field near the Titanic wreck.",
               "a debris field was discovered within the search area by a remotely operated vehicle (ROV) near the Titanic wreck",
               "guardian:world/live/2023/jun/22/titanic-sub-live-updates-search-titan-missing-submarine-submersible-rescue-us-coast-guard-latest-news#64946e658f08657ec7320d6c",
               "The Guardian", "2023-06-22T15:55:03Z", "",
               _G + "world/live/2023/jun/22/titanic-sub-live-updates-search-titan-missing-submarine-submersible-rescue-us-coast-guard-latest-news"
               "?page=with:block-64946e658f08657ec7320d6c#block-64946e658f08657ec7320d6c",
               "Debris found; crew likely lost"),
        ],
        "hindsight": "The Titan imploded on its way down on 18 June, killing all five aboard.",
        "known_from": "2023-06-22T15:55:03Z",
    },
}


def answer(question: str, cutoff: date | datetime | str, guard: bool = True,
           event: str = DEFAULT_EVENT) -> AnswerResult:
    """Return canned answers for the event that change with the cutoff."""
    story = STORIES[event]
    limit = parse_cutoff(event, cutoff)
    visible = [dict(s) for s in story["sentences"] if parse_cutoff(event, s["published"]) <= limit]

    groups: list[dict] = []
    for s in visible:
        claim = s.pop("_claim")
        idx = next((i for i, g in enumerate(groups) if g["claim"] == claim), None)
        if idx is None:
            groups.append({"claim": claim, "count": 0, "article_ids": []})
            idx = len(groups) - 1
        groups[idx]["count"] += 1
        groups[idx]["article_ids"].append(s["article_id"])
        s["group"] = idx
        if event == "titanic-1912":
            s["published"] = s["date"]

    removed: list[dict] = []
    if limit < parse_cutoff(event, story["known_from"]):
        # Simulates the model "slipping" with hindsight. The quote check catches it.
        if guard:
            removed.append({"text": story["hindsight"], "reason": _NO_QUOTE})
        else:
            visible.append({"text": story["hindsight"],
                            "quote": "(no matching quote in any article dated on or before the cutoff)",
                            "article_id": "", "paper": "", "date": cutoff_label(event, limit)[:10],
                            "published": cutoff_label(event, limit), "page": "", "url": "", "group": -1})

    return {
        "sentences": visible,
        "removed": removed,
        "groups": groups,
        "retrieved": [s["article_id"] for s in visible if s["article_id"]],
    }
