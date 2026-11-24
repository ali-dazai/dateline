"""Compose an answer from passages: quoted sentences, a claim per passage, and per-paper figures.

The model is shown only the chosen passages. Every sentence it writes carries a verbatim quote and the
article ID it came from; `pipeline.verify` later removes any sentence its source does not support. This
module only asks, retries once if the reply cannot be parsed, and cleans the shape of what comes back.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dateline import llm
from pipeline import prompts
from pipeline.labels import resolve, short_labels

COMPOSE_MAX_TOKENS = 1500
MAX_SENTENCES = 6


class ComposeError(RuntimeError):
    """The model did not return a usable reply."""


class _Shape(ValueError):
    """The reply parsed but is not the expected shape."""


def _render(passages: dict[str, str], articles: dict[str, dict], labels: dict[str, str]) -> str:
    blocks = []
    for label, aid in labels.items():
        meta = articles.get(aid, {})
        blocks.append(f"<<<ARTICLE id={label} paper={meta.get('paper', '')} date={meta.get('date', '')}>>>\n"
                      f"{passages[aid]}\n<<<END>>>")
    return "\n".join(blocks)


def _ask(prompt: str, llm_opts: Mapping | None, **extra: Any) -> dict:
    """One model call; raises ValueError (including JSON errors) for an unusable reply."""
    reply = llm.complete_json(prompt, max_tokens=COMPOSE_MAX_TOKENS, **extra, **(llm_opts or {}))
    if not isinstance(reply, dict) or not isinstance(reply.get("sentences"), list):
        raise _Shape("expected a JSON object whose 'sentences' is a list")
    return reply


def _claim(value: Any) -> str | None:
    """A stripped claim, or None if it is missing, not a string or blank."""
    return value.strip() or None if isinstance(value, str) else None


def _clean_sentences(raw: list, labels: dict[str, str]) -> list[dict]:
    out = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        fields = [item.get("text"), item.get("quote"), resolve(item.get("article_id"), labels)]
        if not all(isinstance(v, str) and v.strip() for v in fields):
            continue
        out.append({"text": fields[0].strip(), "quote": fields[1].strip(), "article_id": fields[2].strip(),
                    "claim": _claim(item.get("claim"))})
        if len(out) == MAX_SENTENCES:
            break
    return out


def _clean_claims(raw: Any, passages: dict[str, str], labels: dict[str, str]) -> dict[str, str | None]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str | None] = {}
    for key, v in raw.items():
        aid = resolve(key, labels)
        if aid in passages and (v is None or isinstance(v, str)):
            out[aid] = _claim(v)
    return out


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            return float(value.replace(",", "").strip())
    except (ValueError, OverflowError):
        return None
    return None


def _clean_figures(raw: Any, passages: dict[str, str], articles: dict[str, dict],
                   labels: dict[str, str]) -> list[dict]:
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        aid = resolve(item.get("article_id"), labels) if isinstance(item, dict) else None
        if not isinstance(aid, str) or aid not in passages:
            continue
        value = _number(item.get("value"))
        if value is None or value != value or value in (float("inf"), float("-inf")):
            continue
        paper = item.get("paper")
        if not isinstance(paper, str) or not paper:
            paper = articles.get(aid, {}).get("paper", "")
        unit, claim = item.get("unit"), item.get("claim")
        out.append({"paper": paper, "article_id": aid, "value": value,
                    "unit": unit if isinstance(unit, str) else "", "claim": _claim(claim)})
    return out


def compose(question: str, moment: str, passages: dict[str, str], articles: dict[str, dict],
            llm_opts: Mapping | None = None) -> dict:
    """Ask the model for quoted sentences, a claim per passage and figures; raise ComposeError on failure."""
    labels = short_labels(passages)
    prompt = prompts.COMPOSE.format(question=question, moment=moment, passages=_render(passages, articles, labels))
    try:
        try:
            reply = _ask(prompt, llm_opts)
        except ValueError as err:  # includes json.JSONDecodeError
            retry = f"{prompt}\n\nYour previous reply could not be parsed: {err}. Reply with JSON only."
            reply = _ask(retry, llm_opts, use_cache=False)
    except ValueError as err:
        raise ComposeError(f"the model's reply could not be parsed: {err}") from err
    except Exception as err:
        raise ComposeError(f"the model call failed: {err}") from err
    return {"sentences": _clean_sentences(reply["sentences"], labels),
            "claims": _clean_claims(reply.get("claims"), passages, labels),
            "figures": _clean_figures(reply.get("figures"), passages, articles, labels)}
