# Dateline front end

Vite + React + TypeScript. One start page and three story pages, each in its own era look, with an animated switch between looks that depends on where you came from and where you are going.

| Story (API `event`) | Look | Picker |
| --- | --- | --- |
| `titanic-1912` | Sunday Plates: 1912 four-colour printing; the day numeral prints plate by plate | Days, April 10 to 20, 1912 |
| `lehman-2008` | Highlighter: photocopies on an ultramarine desk, one highlighter colour per claim | Days, September 10 to 19, 2008 |
| `titan-2023` | Abyss: a lock screen sinking through the water column | Any minute, June 18 to 23, 2023 (Eastern time), plus stops on the time rail |

Each page also shows its preset questions (below the question box) and a "What a plain AI says" button beside the answer.

## Run it

From the repo root, in two terminals:

```powershell
.venv\Scripts\python.exe -m uvicorn api.server:app --reload --port 8000
cd web; npm install; npm run dev        # http://localhost:5173 (proxies /api to port 8000)
```

For the demo, build once and let the API serve the site at http://localhost:8000:

```powershell
cd web; npm run build                   # writes web/dist, which api/server.py mounts at /
```

`DATELINE_ENGINE=fake` (the default) answers from the stand-in engine and the pages say so. Set `DATELINE_ENGINE=real` in `.env` to use the retrieval engine; nothing in the front end changes.

## What it calls

- `GET /api/health`: `engine` decides whether the stand-in note shows.
- `GET /api/events`: ids, titles, ranges, granularity, look.
- `GET /api/range?event=<id>`: per-day (or per-hour UTC) counts; `story` sizes the newspaper bundles (1912, 2008) and the hourly bars on the 2023 time rail.
- `POST /api/answer` with `{event, question, cutoff, guard}`. Day stories send `YYYY-MM-DD`; the 2023 page sends an ISO time with the Eastern offset, e.g. `2023-06-21T14:00:00-04:00`.
- `POST /api/plain` with `{event, question, cutoff}`, only when the reader presses "What a plain AI says": returns `{answer, moment}` (`answer` empty and a `notice` if the model failed), the prompt-only baseline from `eval/baseline.py`: the same model told the moment with no search, the question prefixed with the story it is about ("About the Titanic (April 1912): ...") as Dateline's search knows it; `moment` is the wording it was told, shown in the panel's label. It is shown beside Dateline's answer (under it on narrow screens), labelled as having no sources, clamped to about ten lines with "Read the whole reply", and is cleared when the question or the moment changes. It never holds up Dateline's answer.

The pages use these optional fields when an engine sends them, and fall back when it doesn't:

| Field | Used for | Fallback |
| --- | --- | --- |
| `sentences[].group` | Pen colour / claim per quote | Looked up from `groups[].article_ids` |
| `sentences[].published` | Source time on the 2023 page | `date` |
| `sentences[].context`, `hit` | Source text around the quote, with the quote highlighted | The quote alone |
| `sentences[].match` | "Matched 95% against the OCR text" | Not shown |
| `sentences[].place` | Paper's city in the citation | Not shown |
| `groups[].papers` | Paper names under each claim | Papers of the quoted sentences in that group |
| `chart` | Dot chart of numbers by paper (e.g. the April 16 death tolls) | Not shown |
| `notice` | Shown in the answer area (e.g. "No articles had been published by this moment."); replaces the empty-answer text | The page's own empty-answer text |
| `sentences[].flag` | Quote check off: the sentence is marked as one the check would remove, with the reason | Not marked |
| `groups[].estimated` | The count reads "about N papers" ("1 paper, maybe more") | "N papers" |

`groups[].count` is the number of distinct papers. At most five claims show (the engine sorts them by count); a "Show N more claims" button shows the rest. A sentence with an empty `article_id` is shown as unsourced (only happens with the quote check off).

## Preset questions

`data/presets.json` holds each story's preset questions: `{"<event>": [{"question", "cutoff", "label"?}]}`, with `cutoff` in the page's own format (`YYYY-MM-DD` for the day stories, an ISO time with the Eastern offset for 2023). `src/stories.ts` imports it, so a change needs `npm run build` (the demo launcher rebuilds when the file is newer than the build). Choosing a preset fills the question and moves the day or minute; `label`, if given, is shown instead of the question.

## Where things are

- `src/App.tsx`: loads the stories, keeps one full-screen layer per page, runs the transitions, and keeps the address bar in step (`#titan-2023`; Back works).
- `src/app.css`: transition keyframes. Classes are `in-<look>-from-<look>` on the arriving page and `out-<look>-to-<look>` on the leaving one (looks: `start`, `plates`, `highlighter`, `abyss`).
- `src/StartPage.tsx`, `src/start.css`: the three bands. Band height must match `bandHeight()` (used to grow a band into its page and shrink a page back into its band).
- `src/themes/`: one component and stylesheet per look.
- `src/stories.ts`: each story's question, hook, default moment, and the 2023 time-rail stops and markers; it adds the presets from `data/presets.json`.
- `src/api.ts`: API calls and the fallbacks above.
- `src/Shared.tsx`: the parts every page has (story links, quote-check switch, presets, the plain-AI panel), styled by each look's stylesheet.

Reduced motion: transitions are skipped and page animations are off.
