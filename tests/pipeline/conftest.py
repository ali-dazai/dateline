"""Shared fixtures for the pipeline tests: tiny corpora, their indexes, and a scripted model."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import pytest

from dateline import config, events, llm

DAY_EVENT = "test-day"
MINUTE_EVENT = "test-minute"

_DAY_ROWS = [
    # (date, paper, state, page, headline, text)
    ("1912-04-14", "The Evening Star", "District of Columbia", "3", "SPRING FASHIONS FOR SUNDAY",
     "The ladies of the city were out in numbers on Saturday to view the new spring hats at the "
     "milliners on F street. Straw braid with ostrich plumes is much in favor this season."),
    ("1912-04-15", "The New York Herald", "New York", "1", "TITANIC STRUCK ICEBERG; ALL SAFE",
     "The White Star liner Titanic struck an iceberg on her maiden voyage, but wireless messages say "
     "that all pas- sengers are safe and that the big ship is being towed to Halifax."),
    ("1912-04-15", "The Evening Star", "District of Columbia", "1", "TITANIC HITS ICE, PASSENGERS TAKEN OFF",
     "NEW YORK, April 15. The new liner Titanic, the largest steamship afloat, collided with ice "
     "off the Grand Banks. ALL PASSEN-\nGERS WERE TAKEN OFF in the boats and are on the way to "
     "New York aboard the Carpathia, says a dispatch to the White Star offices."),
    ("1912-04-15", "The Daily Missoulian", "Montana", "2", "Big Liner Hits Berg",
     "Cape Race, N. F.  The Titanic is reported to have struck an iceberg. The "
     "steamer is believed to be unhurt and her people are in no danger."),
    ("1912-04-15", "The Salt Lake Herald", "Utah", "1", "LINER TITANIC IN COLLISION",
     "The Titanic, the greatest ship ever built, is in collision with a berg. The "
     "White Star line says there is no cause for alarm, as the vessel is unsinkable."),
    ("1912-04-15", "The Washington Times", "District of Columbia", "1", "TITANIC PASSENGERS SAFE, SAYS LINE",
     "Vice President Franklin of the White Star line stated this afternoon that he was assured "
     "the Titanic would reach port under her own steam, and that the passengers are safe."),
    ("1912-04-15", "The Daily Missoulian", "Montana", "4", "Local Markets",
     "Wheat held steady at the close. Hogs were a shade lower. Eggs, fresh ranch, 22 cents."),
    ("1912-04-15", "The Bismarck Tribune", "North Dakota", "1", "STEAMER STRIKES ICE",
     "A great ocean liner, the Titanic, has been in a collision with ice, wireless "
     "reports from the North Atlantic state. Other vessels are hurrying to her aid."),
    ("1912-04-16", "The New York Herald", "New York", "1", "TITANIC SINKS; 1,500 LOST",
     "The Titanic sank on Monday morning after striking an iceberg. About 1,500 lost their lives. "
     "Captain Smith went down with his ship. The Carpathia has picked up 675 survivors."),
    ("1912-04-16", "The Evening Star", "District of Columbia", "1", "TITANIC SUNK, HUNDREDS DROWNED",
     "The giant liner Titanic is at the bottom of the Atlantic. Women and children were saved in "
     "the boats, but more than 1,500 lost when she went down. The Carpathia is bringing the "
     "survivors to New York."),
    ("1912-04-16", "The Salt Lake Herald", "Utah", "1", "WORLD HORRIFIED BY TITANIC DISASTER",
     "Capt. E. J. Smith went down with his ship, according to survivors landed by the Carpathia. "
     "The great vessel sank at 2.20 o'clock. Wireless men stuck to their posts."),
    ("1912-04-16", "The Washington Times", "District of Columbia", "2", "WHITE STAR LINE CONFIRMS LOSS",
     "The White Star line confirmed that the Titanic foundered with great loss of life. The "
     "message from the Carpathia gave no list of the saved."),
    ("1912-04-16", "The Bismarck Tribune", "North Dakota", "1", "FEAR FOR LINER'S PEOPLE",
     "Grave fears are felt for the people of the Titanic, which sank, it is feared, after "
     "the collision with the ice. The Virginian and the Carpathia are near the spot."),
    ("1912-04-16", "The Daily Missoulian", "Montana", "1", "Many Perish In Sea Tragedy",
     "More than 1,500 lost is the estimate in the sinking of the Titanic. The stricken "
     "ship went under in two hours."),
    ("1912-04-16", "The Evening Star", "District of Columbia", "5", "BASEBALL SCORES",
     "Washington 4, Boston 2. The Senators won behind the pitching of Johnson. "
     "Philadelphia and New York were postponed on account of rain."),
    ("1912-04-16", "The Salt Lake Herald", "Utah", "3", "Cattle Prices Up",
     "Cattle at the stock yards were quoted a quarter higher. Receipts were light."),
    ("1912-04-16", "The Washington Times", "District of Columbia", "3", "SENATE CONVENES AT NOON",
     "The Senate met at noon and took up the tariff bill. Mr. Smith of Georgia spoke on the "
     "measure for an hour."),
    ("1912-04-16", "The Bismarck Tribune", "North Dakota", "4", "ROAD BOND ELECTION SET",
     "The county commissioners set the road bond election for the first Tuesday in May."),
]

_MINUTE_ROWS = [
    # (available_at, date, headline, text, section)
    ("2023-06-18T16:20:00Z", "2023-06-18", "Search under way for missing Titan submersible",
     "A search is under way in the North Atlantic for a submersible that went missing during a "
     "dive to the wreck of the Titanic.", "world"),
    ("2023-06-19T14:00:00Z", "2023-06-19", "Coast guard says Titan has about 96 hours of air",
     "The US Coast Guard said the Titan submersible has roughly 96 hours of breathable air.", "world"),
    ("2023-06-20T18:30:00Z", "2023-06-20", "Rescue ships arrive at the search area",
     "Canadian and American vessels reached the search area on Tuesday evening.", "world"),
    ("2023-06-21T05:39:00Z", "2023-06-21", "Banging sounds detected in search for Titan",
     "Search aircraft heard banging noises at 30-minute intervals, officials said. The "
     "submersible search continues.", "world"),
    ("2023-06-21T09:15:00Z", "2023-06-21", "Sonar buoys heard noises again",
     "Sonar buoys again picked up noises near the search area, but remotely operated vehicles "
     "found nothing.", "world"),
    ("2023-06-21T10:00:00Z", "2023-06-21", "Hope fades as oxygen supply dwindles",
     "Experts said the oxygen aboard the Titan submersible would run out on Thursday morning.",
     "world"),
    ("2023-06-21T10:01:00Z", "2023-06-21", "French robot dive joins Titan search",
     "A French deep-sea robot was lowered into the water to join the submersible search.", "world"),
    ("2023-06-21T22:00:00Z", "2023-06-21", "Underwater noises not a lead, experts caution",
     "Acoustic experts cautioned that the noises may not come from the submersible.", "world"),
    ("2023-06-22T15:55:00Z", "2023-06-22", "Debris field found near the Titanic wreck",
     "A debris field was discovered on the seabed near the Titanic. The company said the "
     "Titan crew was believed lost in a catastrophic implosion.", "world"),
    ("2023-06-22T19:00:00Z", "2023-06-22", "OceanGate under scrutiny after Titan loss",
     "Questions are being asked about the safety record of OceanGate, the operator of the Titan "
     "submersible.", "business"),
    ("2023-06-23T09:00:00Z", "2023-06-20", "Titan: what we know so far",
     "Explainer, edited on Friday morning: the Titan submersible lost contact on Sunday and "
     "wreckage was found on Thursday.", "world"),
    ("2023-06-18T09:00:00Z", "2023-06-18", "Weekend recipes: a summer salad",
     "Tomatoes, basil and a good olive oil are all you need.", "food"),
]


@dataclass
class Fixture:
    event: str
    corpus: Path
    df: pd.DataFrame
    index: Path


def write_corpus(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def _day_df() -> pd.DataFrame:
    rows = []
    for n, (day, paper, state, page, headline, text) in enumerate(_DAY_ROWS, start=1):
        rows.append({
            "article_id": f"day-{n:02d}", "lccn": f"sn{83000000 + n}", "paper": paper, "state": state,
            "date": day, "edition": "1", "page": page, "headline": headline, "byline": "", "text": text,
            "url": f"https://chroniclingamerica.loc.gov/lccn/sn{83000000 + n}/{day}/ed-1/seq-{page}/",
            "scan_url": f"https://www.loc.gov/item/sn{83000000 + n}/{day}/ed-1/", "event": DAY_EVENT,
        })
    return pd.DataFrame(rows)


def _minute_df() -> pd.DataFrame:
    rows = []
    for n, (stamp, day, headline, text, section) in enumerate(_MINUTE_ROWS, start=1):
        rows.append({
            "article_id": f"min-{n:02d}", "lccn": "", "paper": "The Guardian", "date": day,
            "edition": "", "page": "", "headline": headline, "text": text,
            "published": stamp, "last_modified": stamp, "available_at": stamp,
            "url": f"https://www.theguardian.com/world/2023/jun/{day[-2:]}/item-{n}",
            "section": section, "kind": "block",
        })
    return pd.DataFrame(rows)


def _register(monkeypatch, tmp_path: Path, name: str, spec: dict, df: pd.DataFrame) -> Fixture:
    from pipeline import index

    corpus = write_corpus(df, tmp_path / f"articles_{name}.parquet")
    monkeypatch.setitem(events.EVENTS, name, {**spec, "corpus": str(corpus)})
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    out = index.build(name)
    return Fixture(event=name, corpus=corpus, df=df, index=out)


@pytest.fixture
def day_event(tmp_path, monkeypatch) -> Fixture:
    spec = {"title": "Test day story", "dates": "April 14-16, 1912", "start": "1912-04-14",
            "end": "1912-04-16", "granularity": "day", "look": "sunday-plates",
            "story_terms": ["titanic"], "source": "test"}
    return _register(monkeypatch, tmp_path, DAY_EVENT, spec, _day_df())


@pytest.fixture
def minute_event(tmp_path, monkeypatch) -> Fixture:
    spec = {"title": "Test minute story", "dates": "June 18-23, 2023", "start": "2023-06-18T00:00:00Z",
            "end": "2023-06-23T23:59:59Z", "granularity": "minute", "look": "abyss",
            "story_terms": ["titan", "submersible"], "display_offset_hours": -4,
            "display_tz_label": "EDT", "source": "test"}
    return _register(monkeypatch, tmp_path, MINUTE_EVENT, spec, _minute_df())


def build_subset(fixture: Fixture, keep: Callable[[Any], bool], name: str) -> Path:
    """Build a second index from the rows of the fixture's corpus for which keep(row) is true."""
    from pipeline import index

    sub = fixture.df[[bool(keep(row)) for _, row in fixture.df.iterrows()]].reset_index(drop=True)
    corpus = write_corpus(sub, fixture.corpus.with_name(f"subset_{name}.parquet"))
    return index.build(fixture.event, corpus=corpus, out=fixture.index.with_name(f"{name}.sqlite"))


TASKS = ("terms", "rerank", "compose", "classify")


@dataclass
class StubLLM:
    replies: dict[str, Any] = field(default_factory=dict)
    calls: list[tuple[str, str]] = field(default_factory=list)

    def __call__(self, prompt: str, **_ignored: Any) -> str:
        first = prompt.lstrip().splitlines()[0].strip() if prompt.strip() else ""
        task = first.split(":", 1)[1].strip() if first.startswith("TASK:") else ""
        if task not in TASKS:
            raise AssertionError(f"stub_llm: prompt has no routable first line 'TASK: <name>' (got {first!r})")
        self.calls.append((task, prompt))
        reply = self.replies.get(task)
        if reply is None:
            raise AssertionError(f"stub_llm: no reply scripted for task {task!r}")
        if callable(reply):
            return reply(prompt)
        if not reply:
            raise AssertionError(f"stub_llm: replies for task {task!r} are exhausted")
        item = reply.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def stub_llm(monkeypatch) -> StubLLM:
    stub = StubLLM()
    monkeypatch.setattr(llm, "complete", stub)
    return stub
