# Evaluation

The question: when a model that already knows how an event ended is asked to answer as of an earlier moment, how often does later knowledge leak into its answer, and how much do date-locked retrieval and the quote check reduce that?

Evaluation uses the 1912 corpus only.

## Model, judges and engine

- **Answer model:** Claude Haiku 5.5 through OpenRouter (`openrouter/anthropic/claude-haiku-5.5`), temperature 0, thinking off (`reasoning_effort` `none`), for every setup and every pipeline call.
- **Judges:** the same model (`DATELINE_JUDGE_MODEL`), same settings.
- **Engine for the retrieval setups:** the retrieval pipeline (`DATELINE_ENGINE=real`, `DATELINE_ENGINE_MODULE=pipeline.engine`, [pipeline-design.md](pipeline-design.md)), run in batch mode (no interactive time limits).

An earlier prompt-only run on Gemini 3.5 Flash Lite (November 2) is superseded and not reported: switching models means re-running every setup, never mixing models across setups.

## Setups

All setups use the same answer model.

| Setup | What it is | Code |
| --- | --- | --- |
| `prompt_only` | The model is told the date and asked to answer as of then; no retrieval | [`eval/baseline.py`](../eval/baseline.py) |
| `rag` | Date-locked retrieval, quote check off (`guard=False`) | engine `answer` |
| `rag_guard` | Date-locked retrieval with the quote check (full Dateline) | engine `answer` |
| `rag_guard_no_expansion` | Ablation: `rag_guard` with search-term expansion off (`DATELINE_TERM_EXPANSION=0`: the question's words and the story's terms only, no terms call) | engine `answer` |
| `rag_guard_no_rerank` | Ablation: `rag_guard` with the re-rank off (`DATELINE_RERANK=0`: the top 10 search hits in BM25 order, the question-words-only top 3 first, no re-rank call) | engine `answer` |

**Setups 2 and 3 share one model answer per row.** The pipeline caches everything the model produced before the quote check runs, so `rag` and `rag_guard` judge the same answer for each row, with the check off and on: the difference between them is the check alone, not a second sample from the model. (In the run below the 30 `rag` answers have 122 sentences; the 30 `rag_guard` answers keep 88 and remove 34, and every `rag_guard` answer is a subset of its `rag` answer.) The ablations change a step before the answer is composed, so they are new model answers, cached under their own keys. `eval.run_experiment` sets both switches explicitly for every retrieval setup (on for the main setups), so a stray environment variable cannot turn a main setup into an ablation, and records them in each row's `steps` column (added after the November 5 run, whose rows leave it empty; that run's environment set neither switch).

## Test set

[`data/testset.csv`](../data/testset.csv): 15 questions about April 1912, asked at one to three cutoffs each (30 rows: Q01 at three, Q15 at one). Each row has:

- `period_answer`: what papers dated on or before the cutoff reported;
- `future_facts`: facts known only after the cutoff, used by the leakage judge;
- `source_article_ids`, `source_urls`: the supporting articles;
- `notes`: the supporting OCR quotes;
- `memorised`: whether the model knows the outcome without a cutoff.

The questions and future facts live in [`data/questions.csv`](../data/questions.csv). [`scripts/build_testset.py`](../scripts/build_testset.py) adds the period answers and refuses to write any row whose quotes are not found in their articles (fuzzy match of at least 90) or whose articles postdate the cutoff.

**How the answer key was built.** The period answers were drafted from the 1912 article text with an AI assistant and accepted only if a script (`scripts/build_testset.py`) finds every supporting quote in an article dated on or before the cutoff; they were never written from model memory of the outcome, so the leakage measurement is not circular.

Known weak rows: Q12 (the corpus has no report of Fenway Park's opening), Q06 (OCR garbles the death counts), Q14 (mixed reports on the Orozco rebellion).

## Tuning and presets

The pipeline's checks and prompts were tuned on the preset candidates (`data/preset_candidates.csv`) and the first 15 test-set rows only (Q01 to Q07), then frozen before the full 30 rows were run. The full 30 include those 15 tuning rows, so results are also reported separately for the 15 held-out rows (Q08 to Q15) that were never used for tuning ([pipeline-design.md](pipeline-design.md), "Real-data check and tuning"). Presets are chosen for the demo, not for evaluation: no preset result is reported as an evaluation result.

## Metrics

- **Leakage rate**: share of answers that state a future fact, computed only over questions the model has memorised (otherwise "no leak" could mean "didn't know"). Judged by [`eval/judges.py`](../eval/judges.py); the judge also sees the period answer, so facts already in print by the cutoff are not counted as hindsight.
- **Empty or fully removed answers, and leakage over non-empty answers**: an empty answer cannot leak, so a setup that removed everything would look leak-free. Each setup reports the share of answers that are empty (nothing composed, or every sentence removed by the check) and its leakage rate over non-empty answers only, beside the overall rate.
- **Period accuracy**: whether the answer matches what the papers said by the cutoff (yes, partial or no).
- **Quote validity** (prompt-only setup): share of the newspaper quotes the model gives that exist in articles dated on or before the cutoff ([`eval/quote_validity.py`](../eval/quote_validity.py)).

Rates come with 95% Wilson confidence intervals. With 26 leakage rows and 30 accuracy rows per setup, the intervals are wide: differences of a few rows between setups are within noise.

## Running it

```powershell
.venv\Scripts\python.exe -m eval.memorisation data\testset.csv   # fills the memorised column
.venv\Scripts\python.exe -m eval.run_experiment                  # the three main setups; resumable
.venv\Scripts\python.exe -m eval.run_experiment --setups rag_guard_no_expansion rag_guard_no_rerank   # ablations
.venv\Scripts\python.exe -m eval.quote_validity                  # needs data\articles.parquet
.venv\Scripts\python.exe -m eval.report                          # results\summary.md, summary.csv, summary_split.csv, chart.png
```

After changing a judge prompt, `python -m eval.rejudge` re-scores the stored answers without generating new ones (the previous file is kept as `results/results_before_rejudge.csv`).

Every LLM response is cached in `cache/`, so reruns cost no quota and results are reproducible. To switch models, change `DATELINE_MODEL`, delete `results/results.csv` and rerun every setup (the three main setups and the two ablations); never mix models across setups.

## Results (Claude Haiku 5.5, November 5, 2026)

All 150 runs (5 setups × 30 rows) passed `validate_result` with no contract problems. Spend for the memorisation check, the five setups and their judges: 409 model calls, $0.11 (many pipeline calls were already in the response cache from the integration runs).

**Memorisation.** Haiku 5.5 knows the outcome of 13 of the 15 questions without a cutoff; Q04 (John Jacob Astor) and Q09 (the Republican primaries) are marked not memorised, so leakage is computed over 26 of the 30 rows.

**All 30 rows.**

| Setup | Leakage | 95% CI | Leakage, non-empty answers | Accurate: yes / partial / no | Accuracy yes (95% CI) | Empty or all removed |
| --- | --- | --- | --- | --- | --- | --- |
| Prompt only | 10/26 (38%) | 22–57% | 10/26 (38%) | 1 / 17 / 12 | 3% (1–17%) | 0/30 |
| RAG | 3/26 (12%) | 4–29% | 3/26 (12%) | 9 / 13 / 8 | 30% (17–48%) | 0/30 |
| RAG + quote check | 4/26 (15%) | 6–34% | 4/25 (16%) | 7 / 10 / 13 | 23% (12–41%) | 1/30 (1 all removed) |
| RAG + quote check, no term expansion | 5/26 (19%) | 9–38% | 5/26 (19%) | 7 / 12 / 11 | 23% (12–41%) | 0/30 |
| RAG + quote check, no re-rank | 3/26 (12%) | 4–29% | 3/26 (12%) | 8 / 12 / 10 | 27% (14–44%) | 0/30 |

**Tuning rows and held-out rows.** The pipeline was tuned on the first 15 rows (Q01 to Q07); Q08 to Q15 were never used for tuning.

| Setup | Rows | Leakage | Leakage, non-empty | Accurate: yes / partial / no | Empty or all removed |
| --- | --- | --- | --- | --- | --- |
| RAG | tuning (Q01–Q07) | 2/13 (15%; CI 4–42%) | 2/13 | 5 / 7 / 3 | 0/15 |
| RAG | held-out (Q08–Q15) | 1/13 (8%; CI 1–33%) | 1/13 | 4 / 6 / 5 | 0/15 |
| RAG + quote check | tuning (Q01–Q07) | 3/13 (23%; CI 8–50%) | 3/12 | 3 / 5 / 7 | 1/15 |
| RAG + quote check | held-out (Q08–Q15) | 1/13 (8%; CI 1–33%) | 1/13 | 4 / 5 / 6 | 0/15 |
| No term expansion | tuning (Q01–Q07) | 4/13 (31%; CI 13–58%) | 4/13 | 4 / 7 / 4 | 0/15 |
| No term expansion | held-out (Q08–Q15) | 1/13 (8%; CI 1–33%) | 1/13 | 3 / 5 / 7 | 0/15 |
| No re-rank | tuning (Q01–Q07) | 2/13 (15%; CI 4–42%) | 2/13 | 5 / 6 / 4 | 0/15 |
| No re-rank | held-out (Q08–Q15) | 1/13 (8%; CI 1–33%) | 1/13 | 3 / 6 / 6 | 0/15 |

The held-out rows do not look worse than the tuning rows, so there is no sign that tuning fitted the evaluation; with 13 rows per half the intervals overlap widely.

**Quote validity (prompt only).** 6 of the 24 newspaper quotes the plain model gave (25%) were found in an article dated on or before the cutoff (`results/quote_validity.csv`); the other 18 were not found (invented, paraphrased beyond the fuzzy match, from papers outside the corpus, or OCR-garbled in the archive: the search first requires the quote's two longest words verbatim, then a partial-ratio match of at least 85). Every quote in a `rag_guard` answer is checked by code against its article.

**Quote check.** Of the 122 sentences in the 30 shared answers, the check removed 34 (28%), and one answer (Q03 at April 15, a single sentence) was removed entirely. The ablations removed 37 of 132 (no term expansion) and 40 of 117 (no re-rank).

**Ablations.** Turning off search-term expansion or the re-rank does not change leakage or accuracy beyond noise (5/26 and 3/26 leaks against 4/26; 7 and 8 accurate against 7). Term expansion was designed so the model's added words carry little weight and cannot add names or numbers; this result is consistent with it neither helping nor steering the search much on these questions. The re-rank's selection likewise did not change the leak rate. Both ablations make one model call fewer per question.

**Reading the numbers.** Date-locked retrieval cuts the judged leakage from 38% (prompt only) to 12–15%, and raises full accuracy from 3% to 23–30%; the intervals still overlap (leakage 22–57% against 4–29%), so with 26 rows the reduction is suggestive rather than conclusive (Fisher's exact test, 10/26 against 3/26: p ≈ 0.05). The quote check does not lower the judged leakage further here (3/26 off, 4/26 on). Because setups 2 and 3 judge the same answer, any leak flagged with the check on is in a sentence also present with the check off, so the extra flag (Q06 at April 16) is a judge inconsistency: the judge flagged a figure printed in a period paper ("1850 people ... drowned", from the Bemidji Daily Pioneer) in the shorter answer and not in the longer one. Read by hand, every leak flag on the guarded setups (and the flagged words in the `rag` answers) is in a sentence whose quote the check found in an article dated on or before the cutoff (early counts of 866 or 650 survivors, an early death estimate, a paper's report that Captain Smith was "a victim of one of those strange fatalities of the sea", a paper's report that the Red Sox "opened their new park yesterday"). On reading, several look like period reports the judge may have taken for later facts. The prompt-only flags include clear hindsight (Ida Straus refusing a lifeboat, in an April 16 answer; the ship "sinking" and "gone down", in an April 14 answer). Lower accuracy with the check on (7 against 9 "yes") likely reflects removed sentences: some honest sentences are removed with the unsupported ones ([pipeline-design.md](pipeline-design.md), "Limitations").

## Judge caveats

- The leakage judge's flags were unreliable on several rows read by hand during preset selection and here: it flags figures and phrases that period papers printed, and it judged the same sentence differently in two answers. It can also miss leaks, so its rates are neither an upper nor a lower bound on real leakage.
- The accuracy judge gives "partial" often (17 of 30 prompt-only answers); the prompt-only answers are long and mix period reports with general statements, which the judge grades as partly matching.

