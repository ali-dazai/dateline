# Engine interface

Status: version 2, proposed 2026-11-02. The web app, the API and the evaluation call the engine only through this interface, so the engine can change freely behind it.

## Function

```python
answer(question: str, cutoff: date | datetime, guard: bool = True,
       event: str = "titanic-1912") -> AnswerResult
```

An engine may accept extra keyword-only arguments; `pipeline.engine.answer` takes `interactive=True` for answers to the web app (short model time limits).

- `question`: the user's question, as typed.
- `cutoff`: for day stories (`titanic-1912`, `lehman-2008`) a `date`, meaning the end of that day; for the minute story (`titan-2023`) a timezone-aware UTC `datetime`.
- `guard`: run the quote check. With `guard=False` unverified sentences stay in the answer (used to show what the check catches, and as an experiment setup).
- `event`: which story's corpus to search. Stories are defined in [`dateline/events.py`](../dateline/events.py).

An engine written before version 2 (no `event` argument) still works for 1912; [`dateline/engine.py`](../dateline/engine.py) wraps it and returns HTTP 501 for the other stories.

## Result

```python
{
  "sentences": [{
      "text": str,          # the sentence shown to the user
      "quote": str,         # verbatim quote from the article that backs it
      "article_id": str,    # empty if the sentence is unsourced (guard=False only)
      "paper": str,
      "date": str,          # YYYY-MM-DD, on or before the cutoff
      "page": str,
      "url": str,           # 1912: https://www.loc.gov/item/{lccn}/{date}/ed-{edition}/
      "published": str,     # optional: ISO 8601 time the item counts from (minute story)
      "group": int,         # optional: index into groups, -1 if none
  }],
  "removed":   [{"text": str, "reason": str}],              # sentences the quote check deleted
  "groups":    [{"claim": str, "count": int, "article_ids": [str]}],
  "retrieved": [str],                                       # article IDs that were searched
  "notice": str,                                            # optional: a message the reader must see
}
```

[`validate_result(result, cutoff)`](../dateline/contract.py) returns a list of contract violations (an empty list means valid). It checks required keys, that every cited item is dated on or before the cutoff, that quotes are not empty, and that `group` indexes into `groups`.

### Optional display fields

The web app uses these when an engine provides them and falls back when it does not:

| Field | Used for | Fallback |
| --- | --- | --- |
| `sentences[].context`, `hit` | Source text around the quote, with the quote highlighted | The quote alone |
| `sentences[].match` | OCR match percentage shown under the quote | Not shown |
| `sentences[].place` | Paper's city in the citation | Not shown |
| `sentences[].flag` | With `guard=False`: why the quote check would have removed the sentence; shown like an unsourced sentence | Not marked |
| `groups[].papers` | Paper names under each claim | Papers of the quoted sentences in that group |
| `groups[].estimated` | `true` when the count rests partly on the model's reading of article openings ("about N papers") | Count shown as exact |
| `notice` | A message shown in place of, or above, the answer: no articles published yet, nothing to search for, nothing found, the model failed, every sentence removed, or the story's index not built | Not shown |
| `chart` | `{title, unit, min, max, points: [{label, value, group}]}`, numbers reported by different papers | Not shown |

## Cutoff rules

- Day stories: an article is available if its `date` is on or before the cutoff day.
- Minute story: a Guardian item is available once its `available_at` is at or before the cutoff. The Guardian API serves each item's latest version, so items edited within 30 days of publication count from their last edit; live blogs are split into entries with their own timestamps, and entries never inherit the blog's (later rewritten) headline. See [`scripts/fetch_guardian.py`](../scripts/fetch_guardian.py).

## HTTP API

Served by [`api/server.py`](../api/server.py); interactive docs at `/docs`.

| Endpoint | Returns |
| --- | --- |
| `GET /api/health` | `{"ok": true, "engine": "fake" \| "real"}` |
| `GET /api/events` | Stories: `id, title, dates, start, end, granularity, look, source` |
| `GET /api/range?event=<id>` | Range plus `counts`: `[{bucket, total, story}]` per day, or per UTC hour for the minute story |
| `POST /api/answer` | Body `{question, cutoff, guard, event}`; `cutoff` is `YYYY-MM-DD` or ISO 8601 with an offset. Returns the result above plus `problems` from `validate_result`. Calls the engine with `interactive=True` when it accepts that keyword; an engine exception becomes an empty result with `notice`, never an HTTP 500 (422 for bad input, 501 for an engine without `event` support) |
| `POST /api/plain` | Body `{question, cutoff, event}`, validated like `/api/answer`. Returns `{answer, moment}` (`moment`: the wording the model was told): the prompt-only baseline (`eval/baseline.py`), the same model told the moment (the day phrase for 1912 and 2008, `pipeline.engine.moment()` for 2023) with no search; `{answer: "", notice}` if the model failed |
| `GET /api/results/summary` | Experiment summary rows (after `eval.report` has run) |

## Plugging in the real engine

1. Expose `answer` with the signature above in a module: `pipeline/engine.py` (the retrieval pipeline, [pipeline-design.md](pipeline-design.md)).
2. Set `DATELINE_ENGINE=real` in `.env`, and `DATELINE_ENGINE_MODULE` to the module path: `pipeline.engine` (the default). The pipeline needs its search index first: `python -m pipeline.index`.
3. Check the output:

```python
from datetime import date
from dateline.contract import validate_result
from pipeline.engine import answer

cutoff = date(1912, 4, 15)
print(validate_result(answer("What happened to the Titanic?", cutoff), cutoff))  # []
```

All LLM calls should go through [`dateline/llm.py`](../dateline/llm.py), which handles caching, rate limiting and retries, and should use the model in `DATELINE_MODEL` so the experiment compares setups on the same model.
