"""HTTP API the front end calls. Thin wrapper around the agreed answer() contract (v2, events).

Run:  .venv\\Scripts\\python.exe -m uvicorn api.server:app --reload --port 8000
Docs: http://localhost:8000/docs

Endpoints
  GET  /api/health                  -> {"ok": true, "engine": "fake" | "real"}
  GET  /api/events                  -> [{"id", "title", "dates", "start", "end", "granularity", "look", "source"}]
  GET  /api/range?event=<id>        -> {"event", "min", "max", "granularity",
                                        "counts": [{"bucket", "total", "story"}]}
                                       buckets are days (1912, 2008) or UTC hours (2023)
  POST /api/answer                  -> AnswerResult (see dateline/contract.py) + "problems"
       body: {"question": str, "cutoff": "YYYY-MM-DD" | ISO datetime, "guard": bool,
              "event": "titanic-1912" | "lehman-2008" | "titan-2023"}   (event defaults to titanic-1912)
       An engine that fails returns an empty result with a "notice", never an HTTP 500.
  POST /api/plain                   -> {"answer": str, "moment": str} (+ "notice" if the model failed)
       body: {"question", "cutoff", "event"}; the prompt-only baseline (eval/baseline.py): the same
       model told the moment ("moment" is the wording it was given) and the story the question is about
       ("About the Titanic (April 1912): ..."), with no search and no sources
  GET  /api/results/summary         -> experiment summary rows (after eval.report has run)

If the built front end exists at web/dist, it is served at /.
"""
from __future__ import annotations

import inspect
import logging
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from dateline import config, events, llm
from dateline.contract import validate_result
from dateline.engine import get_answer
from eval import baseline
from pipeline import engine as pipeline_engine
from pipeline import warm
from pipeline.engine import moment

log = logging.getLogger("dateline.api")

NOTICE_ENGINE_FAILED = "The model could not produce an answer. Try again."
NOTICE_PLAIN_FAILED = "The plain model could not answer. Try again."
# Model-call limits for a request someone is waiting on: the engine's own interactive limits.
INTERACTIVE_LIMITS = {"attempts": pipeline_engine.INTERACTIVE_ATTEMPTS,
                      "timeout_s": pipeline_engine.INTERACTIVE_CALL_S,
                      "deadline_s": pipeline_engine.INTERACTIVE_BUDGET_S}

def _engine_cutoff(event: str, cutoff: datetime):
    """The cutoff as engines take it: a date for day stories (as in v1), a UTC datetime for the minute story."""
    return cutoff.date() if events.get(event)["granularity"] == "day" else cutoff


def _warm_presets() -> None:
    """Run every preset and default whose answer is already stored through the engine once.

    The quote check's first pass over an answer takes ~0.3-0.6 s; after it the same answer comes back in
    well under 0.2 s. Answers not stored are skipped, so this never calls a model."""
    engine = get_answer()
    if not _accepts_interactive(engine):
        return
    for item in warm.items():
        try:
            if pipeline_engine.stored(item.question, item.cutoff, item.event):
                cutoff = events.parse_cutoff(item.event, item.cutoff)
                engine(item.question, _engine_cutoff(item.event, cutoff), True, event=item.event, interactive=True)
        except Exception:  # noqa: BLE001 - a failed warm-up only costs speed
            log.exception("warm-up failed for %r at %s (%s)", item.question, item.cutoff, item.event)


def _preload() -> None:
    try:
        llm.preload()
    except Exception:  # noqa: BLE001 - the first model call imports it instead
        log.exception("model library preload failed")


def _warm_start() -> None:
    try:
        _warm_presets()
    except Exception:  # noqa: BLE001 - e.g. an unreadable presets.json; the import below still runs
        log.exception("preset warm-up failed")
    finally:
        _preload()


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Warm up in the background: the presets' quote checks first (so the first click is instant), then the
    model library import (~10 s, which the first new question would otherwise wait for). /api/health, which
    the launcher waits on, answers at once."""
    threading.Thread(target=_warm_start, name="warm-start", daemon=True).start()
    yield


app = FastAPI(title="Dateline API", lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _revalidate_pages(request, call_next):
    """Make the browser re-check index.html on every visit, so a rebuilt web/dist is never hidden behind a
    cached page that points at asset files the rebuild deleted (the hashed assets themselves may be cached)."""
    response = await call_next(request)
    if response.headers.get("content-type", "").startswith("text/html"):
        response.headers["Cache-Control"] = "no-cache"
    return response


class PlainRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    cutoff: str = Field(description="YYYY-MM-DD (whole day) or ISO 8601 datetime (UTC if no offset)")
    event: str = events.DEFAULT_EVENT


class AnswerRequest(PlainRequest):
    guard: bool = True


def _event(event: str) -> dict:
    try:
        return events.get(event)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e


def _cutoff(req: PlainRequest) -> tuple[dict, datetime]:
    """The story and the cutoff as a UTC datetime; 404 for an unknown story, 422 for a bad cutoff."""
    spec = _event(req.event)
    try:
        cutoff = events.parse_cutoff(req.event, req.cutoff)
    except ValueError as e:
        raise HTTPException(422, f"bad cutoff {req.cutoff!r}: use YYYY-MM-DD or an ISO datetime") from e
    if not events.in_range(req.event, cutoff):
        raise HTTPException(422, f"cutoff must be between {spec['start']} and {spec['end']}")
    return spec, cutoff


def _accepts_interactive(fn) -> bool:
    """True when the engine function names an `interactive` parameter (pipeline.engine does)."""
    try:
        return "interactive" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "engine": config.ENGINE}


@app.get("/api/events")
def list_events() -> list[dict]:
    keys = ("title", "dates", "start", "end", "granularity", "look", "source")
    return [{"id": k, **{f: v[f] for f in keys}} for k, v in events.EVENTS.items()]


@app.get("/api/range")
def corpus_range(event: str = events.DEFAULT_EVENT) -> dict:
    spec = _event(event)
    return {"event": event, "min": spec["start"], "max": spec["end"],
            "granularity": spec["granularity"], "counts": events.counts(event)}


@app.post("/api/answer")
def answer(req: AnswerRequest) -> dict:
    _spec, cutoff = _cutoff(req)
    cutoff_arg = _engine_cutoff(req.event, cutoff)
    try:
        engine = get_answer()  # inside the try: a wrong DATELINE_ENGINE_MODULE is a notice, not an HTTP 500
        # Someone is waiting: an engine that supports it uses its short interactive limits.
        extra = {"interactive": True} if _accepts_interactive(engine) else {}
        result = dict(engine(req.question, cutoff_arg, req.guard, event=req.event, **extra))
    except NotImplementedError as e:
        raise HTTPException(501, str(e)) from e
    except Exception:  # the page shows a notice it can act on instead of an error
        log.exception("engine failed for %r at %s (%s)", req.question, req.cutoff, req.event)
        result = {"sentences": [], "removed": [], "groups": [], "retrieved": [], "notice": NOTICE_ENGINE_FAILED}
    result["problems"] = validate_result(result, cutoff)
    return result


@app.post("/api/plain")
def plain(req: PlainRequest) -> dict:
    """What the same model says when only told the moment: the prompt-only baseline, for comparison."""
    _spec, cutoff = _cutoff(req)
    # Day stories send the baseline's own prompt (the end of that day, which is also what moment() says);
    # the minute story is told the minute.
    told = moment(req.event, cutoff)
    try:
        text = baseline.answer_at(req.question, req.event, cutoff, **INTERACTIVE_LIMITS)
    except Exception:
        log.exception("plain model failed for %r at %s (%s)", req.question, req.cutoff, req.event)
        text = ""
    if not text.strip():
        return {"answer": "", "moment": told, "notice": NOTICE_PLAIN_FAILED}
    return {"answer": text, "moment": told}


@app.get("/api/results/summary")
def results_summary() -> list[dict]:
    path = config.RESULTS_DIR / "summary.csv"
    if not path.exists():
        raise HTTPException(404, "No results yet. Run: python -m eval.report")
    return pd.read_csv(path).fillna("").to_dict(orient="records")


_dist = Path(__file__).resolve().parent.parent / "web" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="web")
