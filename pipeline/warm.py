"""Warm the caches for every preset question and each story's default, so the demo never waits on a model.

    python -m pipeline.warm            answer every item with the quote check on and off, ask the plain model,
                                       and say for each whether it came from the cache
    python -m pipeline.warm --check    make no model call at all; exit 1 listing every item that would need one

Items: each story's default question and cutoff (web/src/stories.ts) and the presets (data/presets.json,
which the web app imports). Engine answers use `interactive=False`, so the long batch retries apply and the
classify step always runs: the stored answer is the full one the page then gets in milliseconds. The plain
answer is the one `/api/plain` serves (eval.baseline.answer_at); the call limits it is given are not part
of the cache key, so the page's interactive request finds it.

`--check` replaces `litellm.completion` with a guard that raises (and records the attempt); when the library
has not been imported it is not imported at all, so the check also runs with the network off. It also fails
an item that breaks without reaching the model (an error, an unreadable cached reply, an empty plain answer)
or whose answer is a notice (no index, nothing found, every sentence removed) instead of sentences.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import types
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from dateline import config, events
from eval import baseline
from pipeline import engine

ROOT = Path(__file__).resolve().parent.parent
PRESETS = ROOT / "data" / "presets.json"
STORIES = ROOT / "web" / "src" / "stories.ts"

_STORY = re.compile(r"^\s*'([a-z0-9-]+)':\s*\{", re.MULTILINE)
_FIELD = r"^\s*{name}:\s*'((?:[^'\\]|\\.)*)'"


@dataclass(frozen=True)
class Item:
    event: str
    question: str
    cutoff: str
    kind: str  # "default" or "preset"


class ModelCallBlocked(RuntimeError):
    """Raised by the --check guard in place of a model call."""


def _unquote(text: str) -> str:
    return re.sub(r"\\(.)", r"\1", text)


def defaults(path: Path = STORIES) -> list[Item]:
    """Each story's opening question and cutoff, read from the web app's story file."""
    text = path.read_text(encoding="utf-8")
    starts = list(_STORY.finditer(text))
    out = []
    for n, match in enumerate(starts):
        block = text[match.end():starts[n + 1].start() if n + 1 < len(starts) else len(text)]
        question = re.search(_FIELD.format(name="question"), block, re.MULTILINE)
        cutoff = re.search(_FIELD.format(name="default_cutoff"), block, re.MULTILINE)
        if question and cutoff:
            out.append(Item(match.group(1), _unquote(question.group(1)), _unquote(cutoff.group(1)), "default"))
    return out


def items(presets: Path = PRESETS, stories: Path = STORIES) -> list[Item]:
    """Per story: the default first, then its presets; an item listed twice is warmed once."""
    by_story: dict[str, list[Item]] = {}
    for item in defaults(stories):
        by_story.setdefault(item.event, []).append(item)
    for event, entries in json.loads(presets.read_text(encoding="utf-8")).items():
        for entry in entries:
            by_story.setdefault(event, []).append(Item(event, entry["question"], entry["cutoff"], "preset"))
    out, seen = [], set()
    for entries in by_story.values():
        for item in entries:
            key = (item.event, engine.normalise_question(item.question),
                   events.parse_cutoff(item.event, item.cutoff))
            if key not in seen:
                seen.add(key)
                out.append(item)
    return out


def answer_stored(item: Item) -> bool:
    """True when the engine's answer cache holds this item's model answer (pipeline.engine.stored)."""
    return engine.stored(item.question, item.cutoff, item.event)


@contextmanager
def _model_calls(block: bool) -> Iterator[list[int]]:
    """Count calls reaching litellm.completion; with `block`, refuse them (and never import litellm)."""
    calls = [0]
    added = False
    if block:
        module = sys.modules.get("litellm")
        if module is None:
            module, added = types.ModuleType("litellm"), True
            sys.modules["litellm"] = module
    else:
        import litellm as module  # noqa: PLC0415 - the slow import only when calls may be made
    had = "completion" in vars(module)
    original = vars(module).get("completion")
    real = getattr(module, "completion", None)

    def guard(*_args, **_kwargs):
        calls[0] += 1
        raise ModelCallBlocked("model call blocked by pipeline.warm --check")

    def counting(*args, **kwargs):
        calls[0] += 1
        return real(*args, **kwargs)

    module.completion = guard if block else counting
    try:
        yield calls
    finally:
        if added:
            sys.modules.pop("litellm", None)
        elif had:
            module.completion = original
        else:
            del module.completion


def _plain(item: Item) -> str:
    return baseline.answer_at(item.question, item.event, item.cutoff)


def _engine(item: Item, guard: bool) -> dict:
    return dict(engine.answer(item.question, item.cutoff, guard, item.event, interactive=False))


def run(todo: list[Item], *, check: bool = False, out: Callable[[str], None] = print) -> list[str]:
    """Warm (or, with `check`, only test) every item; returns the problems found, one line each."""
    problems: list[str] = []
    saved_interval = config.MIN_INTERVAL_S
    if check:
        config.MIN_INTERVAL_S = 0.0  # no call is really made, so nothing to space out
    try:
        with _model_calls(block=check) as calls:
            for n, item in enumerate(todo, 1):
                parts, needs, notes = [], [], []
                steps: list[tuple[str, Callable[[], object]]] = [
                    ("guard on", lambda: _engine(item, True)),
                    ("guard off", lambda: _engine(item, False)),
                    ("plain", lambda: _plain(item)),
                ]
                broken: list[str] = []
                for name, step in steps:
                    before = calls[0]
                    try:
                        value = step()
                        error = ""
                    except Exception as exc:  # noqa: BLE001 - one failed item must not stop the rest
                        value, error = None, f"{type(exc).__name__}"
                    made = calls[0] - before
                    if made:
                        needs.append(name)
                        parts.append(f"{name}: {'NEEDS MODEL' if check else 'new'} ({made} call{'s' * (made > 1)})")
                    else:
                        parts.append(f"{name}: {'error ' + error if error else 'hit'}")
                        if error:  # broke without reaching the model: the page would fail the same way
                            broken.append(f"{name} raised {error}")
                    if name == "guard on" and isinstance(value, dict):
                        notes.append(f"{len(value['sentences'])} sentences, {len(value['groups'])} groups"
                                     + (f", notice: {value['notice']}" if value.get("notice") else ""))
                        if value.get("notice") and not made:  # a preset must show an answer, not a notice
                            broken.append(f"notice: {value['notice']}")
                    if name == "plain" and not error and not made and not (isinstance(value, str) and value.strip()):
                        broken.append("plain answer empty")
                where = f"{item.event} {item.cutoff} {item.question!r} ({item.kind})"
                if check and needs:
                    problems.append(f"{where} would call the model: {', '.join(needs)}")
                if broken:
                    problems.append(f"{where}: {'; '.join(broken)}")
                elif not check and not answer_stored(item):
                    problems.append(f"{where}: engine answer not stored (a model call failed)")
                out(f"[{n}/{len(todo)}] {item.event} {item.cutoff} {item.kind} {item.question!r}\n    "
                    + " | ".join(parts) + (f"   ({notes[0]})" if notes else ""))
    finally:
        config.MIN_INTERVAL_S = saved_interval
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="make no model call; exit 1 if any item needs one")
    parser.add_argument("--presets", type=Path, default=PRESETS)
    parser.add_argument("--stories", type=Path, default=STORIES)
    args = parser.parse_args(argv)
    todo = items(args.presets, args.stories)
    print(f"{len(todo)} items, model {config.ANSWER_MODEL}{' (check: no model calls)' if args.check else ''}",
          flush=True)
    problems = run(todo, check=args.check, out=lambda line: print(line, flush=True))
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for line in problems:
            print(f"  {line}")
        return 1
    print("\nall items answered from the caches" if args.check else "\nall items warmed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
