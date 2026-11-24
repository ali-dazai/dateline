# Retrieval pipeline: design

Status: draft for review, 2026-11-05.

This is the design for `pipeline/`, a complete implementation of the engine interface ([engine-interface.md](engine-interface.md)). It answers questions about any story whose articles are in the standard corpus shape, using only articles available at the cutoff, and proves every sentence with a quote that code checks. It is selected with one setting (`DATELINE_ENGINE_MODULE`), and the same engine serves the demo and the evaluation.

## Goals and success criteria

1. **Same interface.** `pipeline.engine.answer(question, cutoff, guard=True, event=...)` returns a result that passes `validate_result` for all three stories, day and minute cutoffs alike. The web app and `eval/` need no changes to use it (`DATELINE_ENGINE=real`, `DATELINE_ENGINE_MODULE=pipeline.engine`).
2. **Strict cutoff.** No text, ranking statistic, label or number produced after the cutoff can reach the answer. Tested as an invariant: results at a cutoff are identical whether or not later articles exist in the index.
3. **Correct demo answers.** The preset questions return period-correct answers with real quotes at their cutoffs; turning the quote check off shows what it removes.
4. **Speed.** A new question answers in at most 15 s (95th percentile); a repeated question in under 200 ms.
5. **Any story.** Nothing is specific to the three demo stories. A new story is one corpus file in the standard shape plus one entry in `dateline/events.py`.
6. **Hostable.** Under 300 MB of memory while serving; no start-up build.
7. **Tested.** Unit tests run in CI without network or corpus; an integration run on the real corpora passes before the demo.

## Non-goals

- Fetching articles on demand for any date (future work: needs live archive access, rate-limit handling and unchecked corpus quality).
- Embedding or semantic search.
- New demo stories (three are enough for the demo).
- Public access control (designed with hosting, separately).

## Components

All in `pipeline/`. Each file has one job and can be tested alone.

| File | Job |
| --- | --- |
| `index.py` | Builds one SQLite FTS5 index per story at `data/index_<event>.sqlite` from the story's corpus file. Command: `python -m pipeline.index [event ...]`. The engine never builds: a missing index, or one older than its corpus, gives a notice naming the command. |
| `search.py` | `search(con, terms, cutoff, k=50)`: date-locked BM25 with statistics computed from articles available at the cutoff. |
| `query.py` | Search terms: the question's own words plus model-suggested synonyms, with added names and numbers removed. |
| `rerank.py` | The model picks the most relevant articles from the top 50. |
| `passages.py` | Trims each chosen article to the passages around the search terms, so the model reads a bounded amount of text. |
| `compose.py` | The model writes the answer: sentences with quote, article ID and claim label; claim per article; numeric figures per paper when the question asks for a number. |
| `verify.py` | The quote check and the hindsight checks (below), plus display fields: `match`, `context`, `hit`, `place`. |
| `groups.py` | Disagreement groups and the optional `chart`, with their labels and values checked. |
| `engine.py` | `answer()`: runs the steps (four model calls for a new question), applies the answer cache, returns the contract result. |
| `prompts.py` | All prompt text in one place. |
| `warm.py` | Runs the preset questions (and their plain-model answers) so the demo answers from cache. |

Model calls go through `dateline/llm.py` (disk cache, retries, pacing). That module gains a provider fallback (below), which the judges also use.

## Story contract

A story is an entry in `dateline/events.py` (`start`, `end`, `granularity` = `day` or `minute`, `corpus`) and a parquet file with:

- Required: `article_id`, `paper`, `date` (YYYY-MM-DD), `headline`, `text`, `url`.
- Optional: `available_at` (ISO 8601 UTC; when the item became readable; defaults to the end of `date`), `page`, `state` (shown as place), `scan_url`.

An article is available at a cutoff if `available_at <= cutoff`, for day and minute stories alike (a day cutoff is the end of that day; Guardian items edited later count from their edit). The three demo corpora already have these columns.

## Answering one question

1. **Answer cache.** Key: event, normalised question, the last article available at the cutoff, the model, the index file and the moment text (see "Models, speed and failures"). The cache holds everything the model produced (the draft, hits, chosen passages, the articles used and the classify assignments); `guard` and the checks are applied after it, so the quote check on and off reuse one model answer; this is also what makes setups 2 and 3 a clean comparison.
2. **Search terms (model call 1).** The question's words (stop words removed) are always searched at full weight. The model adds up to 8 synonyms or period spellings for those words only, searched at weight 0.3, as are the story's own terms. Code drops any added term that is a name (capitalised) or contains a digit and is not in the question. The low weight and the question-words-only ranking used for the kept top 3 (step 4) stop the model steering the evidence towards facts it knows from later; an evaluation ablation measures what is left. Terms are reduced to letters and digits and quoted before they reach FTS5, so no input can break the query.
3. **Search (code).** BM25 over headline and text, restricted to available articles, top 50.
4. **Re-rank (model call 2).** The model sees ID, paper, date, headline and the first 60 words of each candidate and returns up to 10 IDs by relevance (not by what it believes is true). Code drops IDs not among the candidates and always keeps the top 3 results of a question-words-only search, so the model cannot filter the evidence down to reports that match the known outcome.
5. **Passages (code).** Each chosen article is cut to its best windows around search-term hits, at most about 600 tokens per article (about 8k tokens in total).
6. **Compose (model call 3).** The prompt states the moment in words ("the end of April 15, 1912"; "10:00 a.m. EDT, Wednesday, June 21, 2023"), gives the passages between delimiters, and asks for JSON:
   - `sentences`: `{text, quote, article_id, claim}`. Rules: attribute reports to papers ("The Sun reported..."); when papers conflict, state each version and who printed it; say when the papers do not yet report something; never state what happened later; each quote verbatim from its passage, 6 to 40 words.
   - `claims`: for each passage, the claim it supports (short label) or none.
   - `figures` (only for "how many / how much" questions): `{paper, article_id, value, unit, claim}`.
   Parse failure: one retry with the parser error appended; then an error result (below).
7. **Checks (code, when `guard` is on).** A sentence moves to `removed` with a stated reason if:
   - its quote is shorter than 6 words, or matches its passage below the OCR-tolerant threshold (rapidfuzz `partial_ratio` after normalising case, whitespace and line-break hyphens; threshold tuned on the test set, starting at 90);
   - its article was not among the passages given, or is not available at the cutoff;
   - it states a number (digits or number words, normalised) that its cited passage does not contain;
   - it names a person, ship, place or organisation (capitalised, not a common word) that neither its cited passage nor the question contains;
   - it uses content words its cited passage does not contain (beyond the question's words and attribution words such as "reported" or "feared"). This catches paraphrased hindsight such as "went down with his ship" or "will file for bankruptcy", where no name or number gives it away.
   Quotes are matched word by word (each part between ellipses separately), so an inserted "not" fails. Numbers and names are matched with OCR tolerance ("1,5OO", "Carpathla"); paper names, places, dates up to the cutoff and the question's words are always allowed. Every OCR-tolerant comparison first asks one negation-safety predicate, so the tolerance never lets a negated form stand for an un-negated one ("uot able" is not "notable", "actlve" does not support "Inactive"; see "Real-data check and tuning").
   With `guard` off, every sentence stays; a sentence that would have been removed carries `flag` with the reason, and the web app marks it.
8. **Classify, groups and chart (model call 4 and code).** Groups come from the per-passage claims. When compose gave two or more distinct claims, a fourth call (classify) assigns each of the other top-50 candidates to one of those claims, or to none, from its first 60 words, so counts cover all 50 (skipped for web answers once 20 s have passed); counts that rest only on that call are marked estimated ("about N papers"). Counts are distinct papers. A claim label or figure is model-written text, so it gets the same number and name checks against its supporting passages; a label that fails is replaced by the shortest supporting quote, a figure that fails is dropped. Sentences get `group` indexes.
9. **Result.** `sentences`, `removed`, `groups`, `retrieved` (the 50 IDs, plus any kept top-3 hit not among them), optional `chart`, and a new optional top-level `notice` for cases the reader must see: the story's index is not built, no articles available yet, the question has no searchable words, nothing relevant found, the model failed, or every sentence removed. The first three need no model call and are not cached. A run that fell short is not cached either: compose failed, time ran out, a model call raised (a reply that arrives but is malformed counts as an answer), or classify was skipped for time; the next request gets a full run. `notice` is added to [engine-interface.md](engine-interface.md) as an optional field, and the web app shows it on all three story pages.

## Search: date-local BM25

FTS5's built-in `bm25()` uses statistics (number of documents, average length, how many documents contain each word) from the whole table, which includes articles after the cutoff. These statistics change only the order of earlier articles, never which articles can appear, but the cutoff should hold for them too. So:

1. Articles are stored in availability order, so a cutoff is a row number R (the last article available). N and average document length are computed over rows up to R.
2. For each query term (stemmed the same way the index stems text), the term frequency in every available article comes from FTS5's vocabulary table (`fts5vocab`, `instance` mode) restricted to rows up to R. Document frequency is the number of those articles.
3. Every available matching article is scored with standard BM25 (k1 = 1.2, b = 0.75) from those numbers; top 50 returned. Nothing in the score depends on a later article, so the invariant test can demand identical results.

Measured on the demo PC (Windows, Python 3.12, SQLite 3.53.1), November 5, 2026:

- **Build:** `python -m pipeline.index` takes 60 s for 1912 (282,013 articles, 557 MB on disk), 0.5 s for Lehman 2008 (561 items, 5.7 MB) and 0.2 s for Titan 2023 (443 items, 1.5 MB).
- **Exact scoring, 1912, cutoffs April 15 and April 20 (5 runs each):** each of the 30 most frequent non-stop terms alone (up to 57,138 articles and 193,749 occurrences per term) takes 156–411 ms median, 455 ms at worst ("mr" at April 20, every article available). A 12-term question query (the question words of "Were all the passengers of the Titanic saved after the ship struck an iceberg?" at weight 1.0, plus rescued, lost, steamer, liner, survivors at 0.3) takes 190 ms median (245 ms max) at April 15 and 309 ms median (338 ms max) at April 20. Every hit was available at its cutoff.
- **Memory:** process working set 105 MB after all runs (peak 116 MB), of which 86 MB is the interpreter with pandas loaded.
- **Lehman 2008 and Titan 2023:** 1–3 ms per query; results non-empty and all available at the cutoff.

Exact scoring is well under the 1 s limit, so the engine uses it. Two things kept it there: a covering index on `meta(rowid, length)`, so the document count, average length and length lookups do not read the wide metadata rows, and plain tuples rather than `sqlite3.Row` for postings. The pool fallback (`pool=5000`: up to 5,000 of the most recent available articles containing one of the query's three date-locally rarest terms, then the same scoring) measured 209 ms at worst; it stays available but unused, because it can change the order (the third hit of the 12-term query at April 20 differs).

## Models, speed and failures

- **Model:** Claude Haiku 5.5 for every call (pipeline, prompt-only baseline, judges), through OpenRouter (`openrouter/anthropic/claude-haiku-5.5`; ID confirmed at setup). Temperature 0. Fallback: the same model through Anthropic directly when `ANTHROPIC_API_KEY` is set, used after OpenRouter errors persist through retries. Same model either way, so results stay comparable.
- **Pacing:** `DATELINE_MIN_INTERVAL` lowered for the paid model (paid models have no OpenRouter request cap).
- **Thinking off:** OpenRouter turns Claude Haiku 5.5's adaptive thinking on by default; its tokens count against each call's output cap, so on the first real run every capped call ended with an empty reply. Calls to OpenRouter models send `reasoning_effort` `none` (`DATELINE_REASONING_EFFORT`), which is part of the response-cache key; an empty reply is never cached.
- **Article labels:** prompts list articles as 1, 2, 3 and replies are mapped back to article IDs (`pipeline/labels.py`). With the full IDs the model copied them with errors and compose replies hit the 1,500-token cap.
- **Model calls per new question:** four (search terms, re-rank, compose, classify); classify runs only when the answer has two or more claims. A repeated question makes none.
- **Timeouts:** answers for the web app (`answer(..., interactive=True)`) get 2 attempts and 20 s per call, 30 s in all (each call is given what is left of the 30 s, compose's retry included), then a notice; the batch evaluation keeps longer retries. Output caps per call (compose: 1,500 tokens, at most 6 sentences). A lock per question stops duplicate runs.
- **Answer cache key:** event, the question (lower-cased, punctuation removed, whitespace collapsed, stop words kept so "not" changes the key), the last available article, the model, the index file's size and modification time, the moment text, a hash of every `pipeline/*.py` file and the two ablation switches (`DATELINE_TERM_EXPANSION` and `DATELINE_RERANK`, both on by default; turning one off skips the search-terms or the re-rank call, and its answers are stored under their own keys, never mixed with normal ones). Stored as `cache/answers/<sha256>.json`, written atomically. A failed call after retries and fallback returns a result with empty `sentences` and a `notice` explaining the failure; the API never returns a half-built result.
- **Cost:** measured $0.0025 per new question (about 18,600 tokens in and 1,200 out over four calls, 30 test-set rows, November 5, 2026); cached questions cost nothing.

## Preset questions and the plain-model comparison

- Each story gets 4 to 6 preset questions at chosen cutoffs: showcase questions (answers that flip across the cutoff, papers that disagree, sentences the check removes) and curiosity questions a casual visitor would ask.
- Selection by test: for each candidate, the prompt-only answer (the plain model told the date) and the pipeline answer are both scored with the leakage and accuracy judges. A candidate becomes a preset only if the plain model leaks or contradicts the period answer and the pipeline answer is accurate and clean.
- **Plain-model button:** `POST /api/plain` (`{question, cutoff, event}`) returns the prompt-only answer (`eval/baseline.py`'s prompt, cached). Each story page gets a button that shows it beside Dateline's answer.
- Presets live in `data/presets.json` (the web app imports it); `python -m pipeline.warm` pre-computes their answers and plain answers, and `python -m pipeline.warm --check` confirms, with no model call, that every one answers from the cache.

## Testing

- **Unit tests (CI, no network, no corpus):** a synthetic corpus of about 30 articles over three days, including later articles that contain outcome words, and a stub model with scripted replies.
  - Date lock: for many random cutoffs, no returned or cited article is after the cutoff.
  - Date-local statistics: scores at a cutoff are identical with and without the later articles in the index.
  - Query safety: punctuation, quotes, FTS5 operators and empty questions never raise.
  - Term filter: added names and numbers are dropped, synonyms kept.
  - Checks: exact quote, OCR-damaged quote, short quote, wrong article, post-cutoff article, number not in passage (digits and words), unsupported name, all-removed case.
  - Groups and chart: failing labels replaced, failing figures dropped.
  - End to end with the stub: `validate_result` passes for a day story and a minute story; guard off and on share one model answer.
- **Tuning without fitting the evaluation:** thresholds and word lists are tuned on the preset candidates and the first 15 test-set rows, then frozen before the full 30-row evaluation. Reports include the share of empty or fully removed answers and leakage over non-empty answers, so removing everything does not count as success. Presets are demo selection, not evaluation.
- **Integration run (PC, real corpora, Haiku):** all 30 test-set rows with the check on and off pass `validate_result`; preset candidates; latency for new and cached questions; pool-approximation check. `scripts/check_pipeline.py` runs it (below).
- **Evaluation:** prompt-only and memorisation re-run on Haiku 5.5, then RAG and RAG + check through `eval/run_experiment.py`; judges on Haiku 5.5.

### Real-data check and tuning (November 5, 2026)

`scripts/check_pipeline.py` answers each row three times through `answer(..., interactive=True)`: check on (a new answer), check off (the cached model answer) and check on again (a cache hit), and writes counts, removal reasons, flags, notices, contract problems and seconds per step (the engine logs one INFO record per step on the `pipeline` logger) to `results/pipeline_check_<set>.csv`. `--fresh` bypasses the model-response cache so every new answer is a real run.

**Tuning set:** the 30 preset candidates in `data/preset_candidates.csv` (10 per story) and the first 15 rows of `data/testset.csv` (Q01 to Q07), 45 questions. The other 15 test-set rows (Q08 to Q15) were not run until the constants and prompts were frozen; they are the held-out rows. Every removed sentence was read against the passage it cites and judged honest (the passage supports it: the removal is a false removal), unsupported (the passage does not support it, but it states nothing from after the cutoff: a paraphrase from another passage, a misread figure, a comment), absence (a "not yet known" sentence that failed), or hindsight.

| Run | Kept | Removed | Honest | Unsupported | Absence | Hindsight |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tuning set, before tuning | 93 | 78 | 59 | 12 | 7 | 0 |
| Tuning set, after round 1 | 146 | 45 | 39 | 5 | 1 | 0 |
| Tuning set, after round 2 | 157 | 48 | 38 | 6 | 4 | 0 |
| Tuning set, checks of 51d57da | 156 | 51 | 42 | 7 | 2 | 0 |
| Full test set, 51d57da, fresh run: tuning rows Q01-Q07 (15) | 51 | 18 | 14 | 3 | 1 | 0 |
| Full test set, 51d57da, fresh run: held-out rows Q08-Q15 (15) | 49 | 16 | 13 | 3 | 0 | 0 |
| Full test set, f8b1e31 checks, fresh run: tuning rows Q01-Q07 (15) | 42 | 14 | 13 | 1 | 0 | 0 |
| Full test set, f8b1e31 checks, fresh run: held-out rows Q08-Q15 (15) | 45 | 20 | 16 | 2 | 2 | 0 |

Haiku at temperature 0 still varies between runs, so differences between rows of this table are partly the model's. The effect of the last two rounds of check changes was measured on fixed model answers instead. Applying the round-2 checks (3420b8b) and the 51d57da checks to the same stored answers gives identical verdicts on all 207 tuning-set sentences and differs on one of 134 full-run sentences ("succeeded" from OCR "sue ceeded", now removed because OCR letter confusions no longer count across a joined pair). Applying the 51d57da and final (f8b1e31) checks gives identical verdicts on the 207 tuning-set sentences and differs on one of the same 134 full-run sentences ("collided" from OCR "coIlided", now removed because an i/l difference right after a vowel is not accepted), so the full set was run once more with the f8b1e31 checks. The later checks (0c6a7f6: no join with a negation word, plain-word prefix guard for OCR name matches) give identical verdicts to f8b1e31 on all 122 stored sentences of that run, and the single negation-safety predicate described below (6dcf800, then b733b96 on the remaining paths) gives identical verdicts to 0c6a7f6 on the same 122 stored sentences and on the 207 tuning-set sentences, and the final checks (4a53e7e: split-off and damaged contractions, hyphenated prefix names, hyphen look-alikes and stops between a prefix and its word) give identical verdicts to b733b96 on the same 122 and 207 sentences, so the f8b1e31 run's numbers stand for the final checks (4a53e7e). The run's check-on pass had 121 sentences; its first row (Q01 at April 14) passed 20 s and was not cached, so the check-off pass called the model again for it and stored that second answer, which has one sentence more, hence 122 stored sentences.

Before tuning, the main causes were article IDs the model mis-copied (sentences cited "unknown" articles whose quotes were in a given passage), the Guardian's paper name (`theguardian.com`) not allowing "Guardian", OCR damage in quotes ("tho", "tt. tanic", glued "l5athe"), irregular verb forms (fell/fallen), and the model's paraphrase. What was changed, each with a unit test from the real case: short article labels in prompts; a web paper name allows its plain name; the story's own name is always allowed; `likewise`, `relayed`, `roughly`, `approximately`, `toward(s)` and `might` need no source; OCR v/w, misread or glued "the", a split word with one letter misread (never a join that pairs a word with a part it negates: "an armed" is not "unarmed", a standalone "in", "im", "il", "ir" or "non" never joins the word it would negate ("in active" is not "inactive"), a standalone "a" or "an" never joins a following word of 3+ letters ("a typical" is not "atypical", so "a board" no longer supports "aboard"), and a negation word never joins its neighbour unless the result is one of the negation words written as one word, cannot, nothing, nobody or nowhere ("not able" is not "notable"; "can not" is still "cannot")), OCR punctuation ignored in the quote score, split scale words joined ("hun dred"; only hundred, thousand, million and billion, so "it's even" or "for ty" state no number), names of 6+ letters compared after OCR letter confusions; a sentence word supported under OCR letter confusions only by a single source word of the same length, both 6+ letters, not a negating-prefix pair, and only for the confusions the model actually corrected (i/l/1, and o/0), never an i/l difference right after a vowel ("iifeboats" supports "lifeboats"; "bound" never supports "found", "toiled" never "tolled", "failing" never "falling"); irregular past forms support each other and a present form is supported by any form, but a past form never by a present one ("is sinking" still does not support "sank"); a curated derivation table (takeover from taken over, night from tonight, day from today or yesterday); and a compose prompt that builds sentences from the quote's own words, asks for 6-word quotes and a self-check. No fuzzy stem matching was added, and every leak test still passes.

Those join rules, and the negating-prefix guards on quote words, names and content words, were each added for one leak path, and review kept finding the next path of the same class (an OCR-damaged negation word joining through the skeleton, "uot able" read as "notable"; an OCR-damaged base word under a prefix on the name path, "actlve" supporting "Inactive"; the fuzzy name ratio, "constitutionai" supporting "Unconstitutional"). The final checks (6dcf800, completed by b733b96) close the class with one predicate that every OCR-tolerant comparison asks first (`_negation_flip` for a word against a word, `_join_negation_flip` for a word against two pieces joined, `_join_unsafe` for the joined pairs a source offers): the two sides may match only when their negation status agrees on the plain words and again on their OCR skeletons, where a word has negation status when it is a negation word (no, not, nor, never, none, nothing, nobody, nowhere, neither, cannot), exact or under OCR letter confusions ("uot", "n0t", "nof", "nevcr"), or when it starts with a negating prefix (un, in, im, il, ir, non, dis, mis, a, an, or their skeleton forms "nn", "ln", "lm", "ll", "lr", "dls", "mls") and removing the prefix brings it closer to the other side (fewer edits), so "unsafe" is "safe" negated and "lnactive" is "active" negated but "about" is not "abont" negated. For a join, the status of the pieces counts: a negation word among them, exact or damaged, refuses the join unless the joined word is itself one of cannot, nothing, nobody or nowhere; a prefix piece makes the join a negated form, which must then also be what the word is ("un harmed" is "unharmed", never "harmed"; "on successfully" is never "unsuccessfully"); a standalone prefix word ("in", "im", "il", "ir", "non", "a", "an", damaged or not) never joins the word after it. Around the predicate, three alignment rules close the splits: a short piece that forms a negation with its neighbour ("n ot") or a prefix before a word ("un safe", "in active") can no longer be skipped on either side, a quote may not start right after a split-off "un", "dis" or "mis", and a source's word after such a prefix is not a word of its own, nor is any join starting there ("un safe" offers "unsafe", never "safe"; "un sea worthy" offers "unsea" and "unseaworthy", never "seaworthy"); a source's joined pair keeps its pieces, so the name paths judge it with the join rule ("on known" never supports "Unknown"); a prefix split into letters ("u n sinkable") counts as the prefix; a prefix joined by a bare hyphen ("non-combatant", "un-named") is one word on both sides, so "non" is never a standalone word in front of its base; contractions under OCR damage ("wasn t", "wasu't", "wasut", "dldnt") tokenise to their base and "not", judged by skeleton, and so (in 4a53e7e) do the period's split-off "n't" ("could n't", "ca n't") and damage at the apostrophe ("wasn'f", "couldn 't", "was'nt", a backtick or acute accent for the apostrophe); a capitalised name with a hyphenated prefix ("Non-Union") is one name in its joined form; an en dash, a soft hyphen, or a stop or comma after "un", "dis" or "mis" ("un.known") joins the prefix to its word as a hyphen does; the polarity check sees damaged or split negations in a verbatim quote ("was uot able" or "wasn t able" is a negation the sentence may not drop). A parametrised matrix in `tests/pipeline/test_negation_matrix.py` (every negation word and every prefix, under OCR damage on the passage or the quote side, clean, misread, split and both, in the dropping and the inserting direction, on the quote, join, polarity, name and content-word paths, plus three-piece splits, joined name candidates, damaged contractions, hyphenated and letter-split prefixes: 575 cases, 379 of which failed on the checks before 6dcf800; 4a53e7e adds 236 for split-off and damaged contractions, separated prefixes and prefix names, 206 of which failed on b733b96) holds the class closed, next to a positive matrix of real OCR damage that still matches ("Carpathla", "PASSEN-GERS", "un harmed", "dis tance", "can not", "tt. tanic", "1,5OO"). The residuals that remain by design: a standalone prefix word is a word, so "in active" supports "active" (and never "inactive"), and a quote may begin after it; an OCR "au" before a word is treated as a prefix piece, not as the article "an", so "au gust" joins into "august" and "au aerobic" into "anaerobic"; a negated word whose prefix OCR misread ("nn-sinkable", "nn sea worthy") supports only its own spelling, never the clean "unsinkable" (a quote still matches it).

**Frozen constants:** `MATCH_THRESHOLD` 90, `WORD_RATIO` 80, `NAME_RATIO` 85, join ratio 90 (85 for a join of the same length, 6+ letters), OCR letter confusions for sentence words (i/l/1 and o/0 only) and names (full skeleton) only at 6+ letters, `MIN_QUOTE_WORDS` 6, the word lists in `pipeline/verify.py` as of commit 4a53e7e and the prompts in `pipeline/prompts.py` as of commit 3420b8b.

**What is left (final full run, 29 honest removals of 121 sentences):** most are words the model corrected from OCR in its sentence where the OCR form is not a plain letter confusion ("unsinkable" for "unsineable", "inquire" for "in Quire", "cross the channel" for "eros sthe chan mel", "woman" for "wolan"); then quotes under 6 words, synonyms and tense changes ("appeared" for "is the name of", "sent" for "sends"), and quotes holding a negation the sentence leaves out. None of the removals in any run was hindsight; whether hindsight passes the check in kept sentences is measured by the leakage judge, not here.

**Full test set (checks f8b1e31, identical verdicts under 0c6a7f6, 6dcf800, b733b96 and the final 4a53e7e; 30 rows, check on and off, fresh model calls, run 2026-11-05 10:29-10:33 UTC after f8b1e31 was committed at 10:27:51 UTC; `results/pipeline_check_full_r2.csv`):** zero contract problems; every row has at least one sentence or a notice (Q03 at April 15 had every sentence removed and got the notice).

| Rows | New p50 | New p95 | New max | Cached max |
| --- | ---: | ---: | ---: | ---: |
| All 30 | 7.9 s | 10.1 s | 20.3 s | 142 ms |
| Tuning rows Q01-Q07 (15) | 7.9 s | 20.3 s | 20.3 s | 117 ms |
| Held-out rows Q08-Q15 (15) | 7.9 s | 10.1 s | 10.1 s | 142 ms |

The 20.3 s answer is the first question of the process, which pays about 14 s to import LiteLLM in its first model call; with 20 s gone, the interactive limits skipped classify, so that answer was not cached and the check-off call ran it again (7.2 s). Every other check-off call reused the cached model answer. Mean seconds per step for a new answer: terms 1.45 (0.99 without the first question), search 0.46, re-rank 1.12, passages 0.01, compose 3.67, classify 1.50 (24 of 30 answers), checks 0.09.

## Limitations

- Day stories cannot separate morning and evening editions: an April 15, 1912 cutoff includes late editions that reported the sinking. The answer shows both versions and the disagreement view counts them.
- The model chooses quotes and claims; keeping the top BM25 results and showing all groups limits, but does not measure, selection bias.
- The checks remove honest sentences. On the 15 held-out 1912 test-set rows (Q08 to Q15, never used for tuning), 16 of 65 sentences (25%) were removed although their passage supports them, mostly because the model corrected OCR spelling or paraphrased; the 15 tuning rows gave 13 of 56 (23%). None of the 34 removals was hindsight. A removal costs an answer a sentence, not correctness, but some answers come back short or empty.
- The quote check proves a quote exists in a period article, not that the sentence's paraphrase is faithful; the leakage judge covers that.
