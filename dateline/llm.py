"""One function for every LLM call in the project.

- Goes through LiteLLM, so any provider works (Gemini, Claude, OpenAI...).
- Caches every response on disk, keyed by a hash of the request, so re-running
  an experiment never spends quota twice.
- Spaces calls out (rate limit) and retries with exponential backoff on 429s.
- Falls back to config.FALLBACK_MODEL when the requested model keeps failing.
- Logs token usage of every real call to usage.csv beside the cache folder.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config

_lock = threading.Lock()
_last_call = 0.0
_usage_lock = threading.Lock()
DEFAULT_ATTEMPTS = 8
USAGE_COLUMNS = ["time", "model", "prompt_tokens", "completion_tokens", "seconds"]


def _is_retryable(exc: BaseException) -> bool:
    name = type(exc).__name__
    text = str(exc).lower()
    return (
        name in {"RateLimitError", "ServiceUnavailableError", "APIConnectionError", "Timeout", "InternalServerError"}
        or "429" in text
        or "resource_exhausted" in text
        or "overloaded" in text
    )


def _cache_key(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _throttle() -> None:
    global _last_call
    with _lock:
        wait = config.MIN_INTERVAL_S - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()


def _now() -> float:
    return time.monotonic()


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _backoff(attempt: int) -> float:
    """Seconds to wait after the given failed attempt (1-based): 5, 5, 8, 16 ... capped at 90."""
    return min(max(2.0 * 2 ** (attempt - 1), 5.0), 90.0)


def _log_usage(model: str, resp: Any, seconds: float) -> None:
    """Append one row per real model call to usage.csv; never raises."""
    try:
        usage = getattr(resp, "usage", None)
        row = [
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            model,
            getattr(usage, "prompt_tokens", "") if usage is not None else "",
            getattr(usage, "completion_tokens", "") if usage is not None else "",
            round(seconds, 2),
        ]
        path = config.CACHE_DIR.parent / "usage.csv"
        with _usage_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            is_new = not path.exists()
            with path.open("a", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh)
                if is_new:
                    writer.writerow(USAGE_COLUMNS)
                writer.writerow(row)
    except Exception:  # noqa: BLE001 - logging must never fail a call
        pass


def _reasoning(model: str) -> str | None:
    """The reasoning effort sent with a call to `model`: config.REASONING_EFFORT for OpenRouter models."""
    return (config.REASONING_EFFORT or None) if model.startswith("openrouter/") else None


def preload() -> Any:
    """Import LiteLLM now (it takes ~10 s), so the first real call does not pay for it. Returns the module."""
    import litellm  # noqa: PLC0415

    return litellm


def _call_once(model: str, messages: list[dict], temperature: float, json_mode: bool,
               max_tokens: int | None, timeout: float) -> str:
    import litellm  # imported lazily so tests can run without network

    _throttle()
    kwargs: dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature,
                              "timeout": timeout}
    if _reasoning(model):
        kwargs["reasoning_effort"] = _reasoning(model)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    started = _now()
    resp = litellm.completion(**kwargs)
    _log_usage(model, resp, _now() - started)
    return resp.choices[0].message.content or ""


def _retrying(model: str, messages: list[dict], temperature: float, json_mode: bool,
              max_tokens: int | None, attempts: int, deadline_at: float | None, timeout: float) -> str:
    """Call one model, retrying transient errors until attempts or the deadline run out."""
    attempt = 0
    while True:
        attempt += 1
        per_call = timeout if deadline_at is None else max(1.0, min(timeout, deadline_at - _now()))
        try:
            return _call_once(model, messages, temperature, json_mode, max_tokens, per_call)
        except Exception as exc:  # noqa: BLE001
            if not _is_retryable(exc) or attempt >= attempts:
                raise
            wait = _backoff(attempt)
            if deadline_at is not None and _now() + wait >= deadline_at:
                raise
            _sleep(wait)


def _call(model: str, messages: list[dict], temperature: float, json_mode: bool,
          max_tokens: int | None = None, deadline_s: float | None = None,
          attempts: int | None = None, timeout_s: float | None = None) -> str:
    """Call `model` with retries; if it still fails, try config.FALLBACK_MODEL once more."""
    deadline_at = None if deadline_s is None else _now() + deadline_s
    timeout = config.CALL_TIMEOUT_S if timeout_s is None else timeout_s
    try:
        return _retrying(model, messages, temperature, json_mode, max_tokens,
                         attempts or DEFAULT_ATTEMPTS, deadline_at, timeout)
    except Exception:
        fallback = config.FALLBACK_MODEL
        if not fallback or fallback == model:
            raise
        if deadline_at is not None and deadline_at - _now() <= 0:
            raise
        return _retrying(fallback, messages, temperature, json_mode, max_tokens,
                         1 if attempts is not None else DEFAULT_ATTEMPTS, deadline_at, timeout)


def complete(
    prompt: str,
    *,
    system: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    json_mode: bool = False,
    use_cache: bool = True,
    max_tokens: int | None = None,
    deadline_s: float | None = None,
    attempts: int | None = None,
    timeout_s: float | None = None,
) -> str:
    """Return the model's text reply. Cached on disk unless use_cache=False; an empty reply is not cached.

    Defaults keep the batch behaviour (8 attempts, 60 s per call). Interactive callers pass
    attempts, timeout_s and deadline_s (an overall budget in seconds); once those run out the
    fallback model gets one try in what remains of the budget. A reply from the fallback is
    cached under the requested model's key.
    """
    model = model or config.ANSWER_MODEL
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    payload: dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature,
                               "json": json_mode}
    if max_tokens is not None:  # only then, so responses cached before it existed stay valid
        payload["max_tokens"] = max_tokens
    if _reasoning(model):
        payload["reasoning"] = _reasoning(model)
    path = config.CACHE_DIR / f"{_cache_key(payload)}.json"
    if use_cache and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["response"]

    text = _call(model, messages, temperature, json_mode, max_tokens, deadline_s, attempts, timeout_s)
    if not text.strip():  # an empty reply (e.g. cut off by max_tokens) is returned but never cached
        return text

    _store(path, {"request": payload, "response": text})
    return text


def _store(path: Path, record: dict) -> None:
    """Write a cache file atomically (temp file + os.replace), so an interrupted write never leaves a
    truncated file that a later call would read as a reply; a failed write only costs the cache."""
    tmp = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f"{path.stem[:16]}.", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(record, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
        tmp = None
    except OSError:
        pass
    finally:
        if tmp is not None:
            Path(tmp).unlink(missing_ok=True)


def parse_json(text: str) -> dict:
    """Parse a JSON object from a model reply, tolerating ```json fences and extra prose.

    Tried in order: the whole (unfenced) reply; the span from the first "{" to the last "}"; and, when that
    span holds more than one object ("{...}\n\n{...}"), the first complete object starting at the first "{".
    Anything after that object is ignored. Raises ValueError (json.JSONDecodeError) if none parses."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            # Two objects in one reply ("{...}\n\n{...}"): take the first complete one.
            obj, _end = json.JSONDecoder().raw_decode(cleaned, match.start())
            return obj


def complete_json(prompt: str, **kwargs: Any) -> dict:
    """Like complete(), but asks for and parses a JSON object."""
    return parse_json(complete(prompt, json_mode=True, **kwargs))
