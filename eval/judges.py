"""LLM judges for the two headline metrics.

leakage_judge: does the answer state any fact that was only known after the cutoff?
accuracy_judge: does the answer match what the papers said as of the cutoff?
"""
from __future__ import annotations

from datetime import date

from dateline import config, llm
from eval.testset import cutoff_phrase

LEAKAGE_PROMPT = """You are checking an answer for hindsight.

Question: {question}
The answer was supposed to use only information available by {when}.

Facts that were NOT known by then (any of these in the answer counts as hindsight):
{facts}
{period_block}
Answer to check:
<<<
{answer}
>>>

Does the answer state, imply, or hint at any of those facts, or any other information that
was clearly only known after {when}? Saying something was uncertain or feared at the time is
NOT hindsight if newspapers of that date said it. Ignore the answer's tone.
Count only information from AFTER that point. Anything the period report above contains was
already known and is never hindsight, even if it matches a listed fact in part. A claim that is
simply wrong or vague is an accuracy problem, not hindsight: do not flag it.

Reply with JSON only:
{{"leaked": true or false, "facts_found": ["the listed facts that appear"], "evidence": "the exact words in the answer that leak, or empty"}}"""

ACCURACY_PROMPT = """You are grading an answer against an answer key.

Question: {question}
Point in time: {when}
Answer key (what newspapers had reported by then, checked by a person):
<<<
{period_answer}
>>>

Answer to grade:
<<<
{answer}
>>>

Grade only whether the answer's main claim matches the answer key for that point in time.
"yes" = matches; "partial" = matches in part or is vague; "no" = contradicts it or misses it.
Set "refused" to true if the answer declines to answer or says it cannot know.

Reply with JSON only:
{{"accurate": "yes" or "partial" or "no", "refused": true or false, "reason": "one sentence"}}"""


def _true(value: object) -> bool:
    """A JSON verdict as a bool: true/"true"/"yes" -> True; false/"false"/"no"/missing -> False."""
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


def _bullet(facts: list[str]) -> str:
    return "\n".join(f"- {f}" for f in facts) or "- (none listed)"


PERIOD_BLOCK = """
What newspapers had reported by then (already known, so NOT hindsight):
<<<
{period_answer}
>>>
"""


def leakage_judge(question: str, cutoff: date, answer: str, future_facts: list[str],
                  period_answer: str = "", *, when: str | None = None) -> dict:
    """Did the answer use information from after the cutoff?

    `period_answer` (what the papers had reported by the cutoff) stops the judge from
    flagging facts that were already in print, such as an early estimate of the dead.
    `when` replaces the day phrase ("the end of Monday, April 15, 1912") for a minute cutoff.
    """
    if not answer.strip():
        return {"leaked": False, "facts_found": [], "evidence": "", "empty_answer": True}
    period_block = PERIOD_BLOCK.format(period_answer=period_answer) if period_answer.strip() else ""
    out = llm.complete_json(
        LEAKAGE_PROMPT.format(question=question, when=when or cutoff_phrase(cutoff),
                              facts=_bullet(future_facts), period_block=period_block, answer=answer),
        model=config.JUDGE_MODEL,
    )
    return {
        "leaked": _true(out.get("leaked")),
        "facts_found": list(out.get("facts_found") or []),
        "evidence": str(out.get("evidence") or ""),
    }


def accuracy_judge(question: str, cutoff: date, answer: str, period_answer: str, *,
                   when: str | None = None) -> dict:
    if not answer.strip():
        return {"accurate": "no", "refused": True, "reason": "empty answer"}
    out = llm.complete_json(
        ACCURACY_PROMPT.format(question=question, when=when or cutoff_phrase(cutoff),
                               period_answer=period_answer, answer=answer),
        model=config.JUDGE_MODEL,
    )
    grade = str(out.get("accurate", "no")).strip().lower()
    if grade not in {"yes", "partial", "no"}:
        grade = "no"
    return {"accurate": grade, "refused": _true(out.get("refused")), "reason": str(out.get("reason") or "")}
