"""Re-rank search hits with one model call, keeping the strongest plain-search hits regardless."""
from __future__ import annotations

from collections.abc import Callable, Mapping

from dateline import llm
from pipeline import prompts
from pipeline.query import switch_on
from pipeline.labels import resolve, short_labels
from pipeline.search import Hit

RERANK_MAX_TOKENS = 300
LIMIT = 10
RERANK_ENV = "DATELINE_RERANK"


def rerank_on() -> bool:
    """Whether the model re-ranks the search hits (DATELINE_RERANK, default on)."""
    return switch_on(RERANK_ENV)


def _candidates(hits: list[Hit], openings: dict[str, str], labels: dict[str, str]) -> str:
    label_of = {aid: label for label, aid in labels.items()}
    lines = []
    for h in hits:
        opening = " ".join(openings.get(h.article_id, "").split())
        lines.append(f"[{label_of[h.article_id]}] {h.paper}, {h.date} — {h.headline} — {opening}")
    return "\n".join(lines)


def _model_ids(question: str, moment: str, hits: list[Hit], openings: dict[str, str],
               llm_opts: Mapping | None, on_fallback: Callable[[str], None] | None = None) -> list[str] | None:
    labels = short_labels(h.article_id for h in hits)
    try:
        reply = llm.complete_json(
            prompts.RERANK.format(question=question, moment=moment,
                                  candidates=_candidates(hits, openings, labels)),
            max_tokens=RERANK_MAX_TOKENS, **(llm_opts or {}),
        )
    except ValueError:  # a reply arrived but is not JSON: a real, if useless, answer
        return None
    except Exception as err:  # noqa: BLE001 - fall back to search order
        if on_fallback is not None:
            on_fallback(f"rerank call failed: {type(err).__name__}")
        return None
    ids = reply.get("ids") if isinstance(reply, dict) else None
    if not isinstance(ids, list):
        return None
    return [aid for i in ids if (aid := resolve(i, labels)) is not None]


def rerank(question: str, moment: str, hits: list[Hit], openings: dict[str, str], keep_ids: list[str],
           limit: int = LIMIT, llm_opts: Mapping | None = None,
           on_fallback: Callable[[str], None] | None = None, use_model: bool = True) -> list[str]:
    """Article IDs to read: `keep_ids` first, then the model's choices; hits in order if the model fails.

    `use_model=False` makes no model call and takes the hits in search order (the same order as a failed
    call). `on_fallback` is called with a short reason when the model call raised (a malformed reply that
    arrives is not reported)."""
    if not hits:
        return []
    known = {h.article_id for h in hits}
    chosen = _model_ids(question, moment, hits, openings, llm_opts, on_fallback) if use_model else None
    if chosen is None:
        chosen = [h.article_id for h in hits]
    out: list[str] = []
    for aid in [*keep_ids, *chosen]:
        if aid in known and aid not in out:
            out.append(aid)
    return out[:limit]
