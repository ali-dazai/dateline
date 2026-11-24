"""The disagreement view: which papers made which claim, and a chart of figures by paper.

Claim labels and figures are written by the model, so they pass the same checks as answer sentences: a
label with a number, name or content word its sources do not contain is replaced by a quote or headline
from those sources, and a figure whose value is not in its passage is dropped.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Mapping

from dateline import llm
from pipeline import prompts, verify
from pipeline.labels import resolve, short_labels

CLASSIFY_MAX_TOKENS = 600


def _key(label: str) -> str:
    return " ".join(label.split()).lower()


def _clean(label) -> str:
    return " ".join(label.split()) if isinstance(label, str) else ""


def classify(question: str, claims: list[str], openings: dict[str, str],
             llm_opts: Mapping | None = None,
             on_fallback: Callable[[str], None] | None = None) -> dict[str, int | None]:
    """Assign candidate articles to one of `claims` (by index) or to none, from their openings.

    Returns {} without a model call when there are fewer than two distinct claims or no openings, and on
    any failure or malformed reply. `on_fallback` is called with a short reason when the model call
    raised (a malformed reply that arrives is not reported)."""
    if len({_key(c) for c in claims if _clean(c)}) < 2 or not openings:
        return {}
    numbered = "\n".join(f"{i}. {_clean(c)}" for i, c in enumerate(claims))
    labels = short_labels(openings)
    candidates = "\n".join(f"[{label}] {' '.join(str(openings[aid]).split())}" for label, aid in labels.items())
    try:
        reply = llm.complete_json(
            prompts.CLASSIFY.format(question=question, claims=numbered, candidates=candidates),
            max_tokens=CLASSIFY_MAX_TOKENS, **(llm_opts or {}),
        )
    except ValueError:  # a reply arrived but is not JSON: a real, if useless, answer
        return {}
    except Exception as err:  # noqa: BLE001 - groups then rest on the passages alone
        if on_fallback is not None:
            on_fallback(f"classify call failed: {type(err).__name__}")
        return {}
    assign = reply.get("assign") if isinstance(reply, dict) else None
    if not isinstance(assign, dict):
        return {}
    out: dict[str, int | None] = {}
    for key, index in assign.items():
        aid = resolve(key, labels)
        if aid not in openings:
            continue
        if index is None:
            out[aid] = None
        elif isinstance(index, int) and not isinstance(index, bool) and 0 <= index < len(claims):
            out[aid] = index
    return out


def _paper(articles: dict[str, dict], aid: str) -> str:
    return _clean((articles.get(aid) or {}).get("paper"))


def build_groups(claims: dict[str, str | None], articles: dict[str, dict], passages: dict[str, str],
                 sentences: list[dict], question: str, allowed: set[str],
                 classified: dict[str, str] | None = None) -> tuple[list[dict], dict[str, int]]:
    """Contract groups (claim, count, papers, article_ids) sorted by count, and article_id -> group index.

    `claims` are the labels compose wrote for the passages it read; `classified` are labels assigned to
    other candidates from their openings. Labels merge case-insensitively. `count` is the number of
    distinct papers. A group with classified support whose full text lacks some of the label's content
    words is marked `estimated`."""
    merged: dict[str, dict] = {}

    def add(label, aid: str, given: bool) -> None:
        label = _clean(label)
        if not label or aid not in articles:
            return
        g = merged.setdefault(_key(label), {"label": label, "given": [], "classified": []})
        if aid not in g["given"] and aid not in g["classified"]:
            g["given" if given else "classified"].append(aid)

    for aid, label in claims.items():
        add(label, aid, True)
    given_ids = {a for g in merged.values() for a in g["given"]}
    for aid, label in (classified or {}).items():
        if aid not in given_ids:
            add(label, aid, False)

    checked: dict[str, dict] = {}
    for g in merged.values():
        ids = g["given"] + g["classified"]
        if g["given"]:
            source = "\n".join(passages.get(a, "") for a in g["given"])
        else:
            source = "\n".join(str(articles[a].get("text") or "") for a in ids)
        label = g["label"]
        if verify.unsupported(label, source, question, allowed) is not None:
            label = _fallback_label(ids, sentences, articles, passages)
            if not label:
                continue
        target = checked.setdefault(_key(label), {"label": label, "given": [], "classified": []})
        target["given"] += [a for a in g["given"] if a not in target["given"]]
        target["classified"] += [a for a in g["classified"] if a not in target["classified"]]

    result = []
    for g in checked.values():
        ids = g["given"] + g["classified"]
        papers = list(dict.fromkeys(p for a in ids if (p := _paper(articles, a))))
        if not papers:
            continue
        group = {"claim": g["label"], "count": len(papers), "article_ids": ids, "papers": papers}
        words = verify.content_words(g["label"])
        if any(not words <= verify.content_words(str(articles[a].get("text") or ""))
               for a in g["classified"]):
            group["estimated"] = True
        result.append(group)
    result.sort(key=lambda g: (-g["count"], g["claim"]))
    group_of = {a: i for i, g in enumerate(result) for a in g["article_ids"]}
    return result, group_of


def _fallback_label(ids: list[str], sentences: list[dict], articles: dict[str, dict],
                    passages: dict[str, str]) -> str:
    """The shortest quote of a draft sentence citing one of `ids` that is found in that article's passage
    (draft quotes are model-written and not yet checked), else the first article's headline."""
    quotes = [_clean(s.get("quote")) for s in sentences if s.get("article_id") in ids
              and verify.quote_match(_clean(s.get("quote")), passages.get(s.get("article_id"), ""))
              >= verify.MATCH_THRESHOLD]
    quotes = [q for q in quotes if q]
    if quotes:
        return min(quotes, key=len)
    return _clean((articles.get(ids[0]) or {}).get("headline")) if ids else ""


def _integer_string(value: float) -> str:
    return str(int(value)) if value == int(value) else str(value)


def build_chart(figures: list[dict], passages: dict[str, str], articles: dict[str, dict],
                group_of: dict[str, int], question: str) -> dict | None:
    """A chart of one figure per paper, or None when fewer than two checked figures remain.

    A figure is kept only if its article is among the passages and its value is a number its passage
    states. Title and unit are built here, not taken from the model."""
    points, units, seen = [], Counter(), set()
    for fig in figures:
        aid = fig.get("article_id")
        try:
            value = float(fig.get("value"))
        except (TypeError, ValueError):
            continue
        if aid not in passages or not math.isfinite(value):
            continue
        if _integer_string(value) not in verify.numbers(passages[aid]):
            continue
        paper = _paper(articles, aid) or _clean(fig.get("paper"))
        if not paper or paper in seen:
            continue
        seen.add(paper)
        unit = fig.get("unit")
        if isinstance(unit, str) and unit.strip():
            units[unit.strip()] += 1
        points.append({"label": paper, "value": value, "group": group_of.get(aid, -1)})
    if len(points) < 2:
        return None
    values = [p["value"] for p in points]
    low, high = min(values), max(values)
    if low == high:
        low, high = (low - 1, high + 1) if low == 0 else sorted((low * 0.9, high * 1.1))
    return {"title": _title(question), "unit": units.most_common(1)[0][0] if units else "",
            "min": low, "max": high, "points": points}


def _title(question: str) -> str:
    text = re.sub(r"[\s?]+$", "", " ".join(str(question or "").split()))
    return f"{text[:1].upper()}{text[1:]}, by paper" if text else "Figures by paper"
