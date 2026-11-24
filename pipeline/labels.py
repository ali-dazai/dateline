"""Short labels that stand in for article IDs in prompts.

Article IDs are long ("guardian:uk-news/live/2023/jun/20/...#64924a488f0890e11a2be8b8",
"1912-04-16_p1_sn92053934_..."). The model copies them with errors (a dropped path segment, a wrong day,
p1 for p2) and spends many output tokens on them, so prompts list articles as 1, 2, 3 ... and replies are
mapped back. A reply that gives a real ID is accepted as well; anything else is returned unchanged, so
the caller's own unknown-ID handling still applies.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def short_labels(ids: Iterable[str]) -> dict[str, str]:
    """{"1": first id, "2": second id, ...} in the given order, duplicates skipped."""
    out: dict[str, str] = {}
    for aid in ids:
        if aid not in out.values():
            out[str(len(out) + 1)] = aid
    return out


def resolve(value: Any, labels: dict[str, str]) -> str | None:
    """The article ID a reply value names: a label (as text or a number) maps to its ID; any other string
    is returned stripped; anything else gives None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        value = str(value)
    if not isinstance(value, str):
        return None
    text = value.strip()
    key = text.strip("[]").strip()
    return labels.get(key, text)
