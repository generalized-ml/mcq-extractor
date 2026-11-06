# PDF-to-MCQ evaluation

Compare extracted MCQs against a golden JSON using exact word-prefix matching
and local sentence embeddings. No LLM API calls are made.

Run from the workspace root, using the Python environment where you installed the dependencies:

```bash
python -m pip install -e ".[eval]"

python -m evaluation \
  --mcq-json dataset/jsons/2021/mcq_with_llm/2021_mcqs_with_llm.json \
  --golden-json dataset/golden/neet_2021/2021_mcqs.json
```

Use `.venv/bin/python` instead of `python` to select the project's existing
environment. The first evaluation downloads `sentence-transformers/all-MiniLM-L6-v2`;
subsequent runs reuse the local model cache. `--local-files-only` prevents downloads.
Use `--model /path/to/local/model` to choose an existing SentenceTransformer model.
Use `--model-revision COMMIT` to pin a model revision.

## Input and sequence

Both inputs accept a JSON array or an object containing a `questions` array.
Question text can be `question_text`, `text`, or `question`. Options can be a
list of strings or objects with `label` and `text`. Labels `1`–`4` and `A`–`D`
identify the four option slots. Without labels, array order defines the slots.

The sixth field, called `solution` in score columns, reads **`correct_answer`**.
String and numeric answers are supported. If that key is absent, it accepts
`solution`, `solution_text`, or `explanation`. It does not solve questions or
infer answers. Labels are compared as written: `A` is not converted into `1`.

Questions are compared **by exact ID**, using `question_number` and falling back
to `question_id` if the number is absent/empty. Numeric IDs are converted to
strings and surrounding whitespace is removed. Array order does not affect
matching. Output rows follow golden dataset order.

If a golden ID has no prediction, every field and both weighted totals score
zero for that row. Later questions still match their own IDs. Predictions with
no ID cannot match and count as extras, as do IDs outside the golden set.
Golden questions must have IDs. Duplicate IDs in either dataset raise an error
to avoid choosing an arbitrary question. `review_candidates` are not inserted.

## Scores

For each question, compare these six fields independently:

| Field | Weight |
| --- | --- |
| Question | 0.3 |
| Option 1 | 0.15 |
| Option 2 | 0.15 |
| Option 3 | 0.15 |
| Option 4 | 0.15 |
| Solution (`correct_answer`) | 0.1 |

**Keyword:** split each field with Python's `str.split()` (whitespace), compare
tokens in order, and stop at the first mismatch. Score = matching prefix length
/ golden token count. Case, punctuation, and formula notation remain significant.
Example: golden `one two three four`, extracted `one WRONG three four` → `0.25`.
An extracted field containing the entire golden prefix followed by extra words
scores `1.0` under this requested metric; word counts expose the extra content.

**Semantic:** compute sentence embeddings locally and compare each corresponding
field using cosine similarity. Clip negative similarities to zero, giving scores
in `[0, 1]`. Long fields are split into model-sized token chunks, whose embeddings
are pooled using token-count weights. Repeated texts are encoded once per run.
Reference: https://www.sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html

Both per-question totals are the weighted sum of their six field scores.
Experiment totals are the mean over **all golden questions**. Missing predictions
score zero; extra predictions are counted separately and are not added to the
golden denominator. Missing/empty fields score zero even when both sides are
empty, and weights are never redistributed. Duplicate predicted option labels
are reported and the ambiguous option slot scores zero; malformed golden
option labels raise an error instead of being silently repaired.

`aggregate_score` is the arithmetic mean of `keyword_score` and
`semantic_score`, at both question and experiment level.

These metrics evaluate **text**, not visual correctness. Image-only fields with
no text score zero; image paths/references are not compared. A correct answer
label is included in semantic scoring as requested, but embedding similarity of
short labels is not a measure of answer correctness. Use its keyword field score
to check exact label agreement. Likewise, semantic similarity can miss changed
numbers, negation, or mathematical symbols; keep both metrics visible.

## Output

Results default to `results`, independent of working directory:

- `experiment.csv`: one row per experiment with overall scores, six field scores
  for each metric, missing/extra counts, and input/model identifiers.

Evaluation now writes only `experiment.csv`, appending one summary row per run.
Per-question CSVs, metadata JSONs, and per-call CSVs are no longer generated.
Existing older reports are left untouched. Detailed LLM calls remain in the input
pipeline JSON.

## Runtime per method

Each experiment row also records the extraction method's elapsed time:

- `method_wall_seconds`: recorded PDF pipeline wall time, in seconds. For
  `--mcq-root`, this is the sum of available per-PDF durations, not the wall time
  of a job that might have processed several PDFs concurrently.
- `method_timed_pdfs` / `method_missing_time_pdfs`: timing coverage across PDFs.
- `method_unfinished_time_pdfs`: PDFs with an interrupted/unfinalized timing
  session; their recorded duration is a subtotal through the last checkpoint.
- `method_incomplete_history_pdfs`: PDFs first timed while resuming older output,
  where work before timing was added cannot be reconstructed.
- `method_runtime_scope`: identifies what was measured.
- `evaluation_model_loading_seconds`, `evaluation_scoring_seconds`, and
  `evaluation_wall_seconds`: evaluation durations, kept separate from extraction.
  Scoring includes embedding generation. Evaluation wall time includes input
  loading and model loading, ending before writing the CSV.

The original, Paddle, native HF, and crop/vision entry points save `runtime.json`
beside each PDF's `pipeline.json` or final JSON. Elapsed time covers preparation,
model loading, OCR/extraction, review, and checkpoint work performed during each
invocation. Reused crop/conversion work is excluded. The timer stops before any
automatic evaluation. Repeated checkpoint snapshots update the same session;
resume creates another session. Durations add active invocation time without
including idle time between commands. Parallel OCR measures elapsed wall time,
so concurrent question times are not added together.

The evaluator reads these sidecars automatically for each method and each year.
It never estimates wall time from summed question times or file timestamps.
Historical outputs without a sidecar have blank durations; partial coverage
produces a subtotal with explicit coverage counts. A bounded run records only
work completed/attempted so far, not an estimate for the whole PDF.

When adding timing columns to an otherwise compatible `experiment.csv`, the
writer preserves existing rows and scores and leaves their timing cells blank.
Other incompatible schema changes still require a new output directory.

## LLM costs

Evaluation automatically reads the extraction costs already saved in the MCQ
JSON. Its cost metric is **average recorded USD per PDF**, rather than the total
cost of all PDFs. `experiment.csv` reports `llm_cost_per_pdf_usd`,
`llm_cost_pdf_count` (the denominator), `llm_missing_cost_pdfs`, `llm_cost_currency`,
`llm_recorded_calls`, `llm_priced_calls`, `llm_missing_cost_calls`, and
`llm_prompt_tokens`, `llm_completion_tokens`, and `llm_total_tokens`. The cost summary is also printed in the CLI.
No additional argument, API key, or LLM call is required to read costs.

For one PDF, cost/PDF is that PDF's recorded cost. For combined years, add the
recorded costs and divide by the number of PDFs with reported costs. For example,
$0.10 and $0.30 across two PDFs gives $0.20/PDF. A third PDF with unknown cost is
excluded from the denominator and counted in `llm_missing_cost_pdfs`; it is never
treated as free. The chart uses the same per-PDF metric and labels the USD axis
accordingly. Partial per-call cost coverage still produces a marked subtotal.

Costs use OpenRouter's saved `usage.cost` value; see its
[usage accounting documentation](https://github.com/OpenRouterTeam/docs/blob/main/cookbook/administration/usage-accounting.mdx)
and [USD credit denomination](https://openrouter.ai/support).
The evaluator prefers `hybrid.results` so that usage copied into question-level
`llm` fields is not counted twice. Without that checkpoint, it reads `llm` from
both `questions` and `review_candidates`. Response IDs are deduplicated.
Costs cover the entire recorded extraction, including incomplete candidates
and questions absent from the golden set.

Missing or invalid costs remain null (blank in CSV), rather than being treated
as free. A reported zero is retained. `llm_missing_cost_calls` exposes missing
per-call records; cost/PDF uses each priced PDF's recorded subtotal. Token totals
likewise include only reported values. Historical outputs
do not retain all failed or retried HTTP attempts, so these reports cannot
reconstruct their charges or serve as a complete billing ledger. Evaluation
itself uses local embeddings and makes no LLM requests.

Compatible CSVs are upgraded when a new experiment is saved. Historical totals
move to `legacy_llm_cost_usd`; their per-PDF cells remain blank because the PDF
denominator was not recorded. Old scores are preserved and are not recomputed.
Charts show historical rows without per-PDF values as unknown. Other incompatible
schema changes still require a new `--output-dir`.

Scores are decimal values, not percentages. Omit `--experiment-number` to use
the next number automatically. Existing experiment numbers cannot be overwritten.
Use `--output-dir PATH` for another results folder.

Earlier evaluator versions used positional alignment. Check each experiment's
metadata for its alignment rule; positional scores should not be compared
directly with scores from the current ID-based alignment.

## Offline tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_evaluate.py'
```

Tests use explicit fake vectors to verify scoring and CSV logic without a model
download; actual evaluation always uses sentence embeddings.

## Shared experiments

The implementation lives in `src/evaluation`. All pipelines append to the same
`results/experiment.csv` by default. `experiment_name` identifies `pdf_to_json`,
`pdf_to_json_paddle`, or `pdf_to_json_my`; `experiment_number` uniquely identifies
each run across pipelines. Input paths and hashes distinguish papers and outputs.
Specify the label explicitly when evaluating a renamed input:

```bash
PYTHONPATH=src .venv/bin/python -m evaluation \
  --experiment-name pdf_to_json_paddle \
  --mcq-json outputs/json_crops_paddle/2021/final_mcqs.json \
  --golden-json dataset/golden/neet_2021/2021_mcqs.json \
  --device cpu
```

Use `--experiment-name pdf_to_json` for the original pipeline. Omit `--output-dir`
to keep all runs in the shared file. Existing two experiment summaries were
migrated without recomputing scores; original reports are preserved in
`outputs/evaluation_legacy_reports.zip`. The old module entry point remains a
compatibility wrapper.

The evaluator can infer the renamed golden path from a year in `--mcq-json`.
For example, a 2021 output uses `dataset/golden/neet_2021/2021_mcqs.json`.
Pass `--golden-json` to override this choice.

Plot the shared experiments with:

```bash
PYTHONPATH=src .venv/bin/python -m evaluation.plot_experiments
```

This saves `results/experiments.png`. Scores use the left axis and recorded LLM
API cost per PDF uses a separate USD/PDF axis. Unknown costs are shown as unknown; partial
costs are marked with an asterisk. Use `--output path.svg` for a vector chart.

## One row for all years

Use `--mcq-root` to combine 2021–2025 into one experiment row. Question IDs are
scoped by year before matching, so the same question number in two papers cannot
collide. A missing yearly prediction file contributes zero predictions for that
paper and lowers the aggregate scores; the CLI prints coverage for each year.
Recorded LLM costs are averaged across PDFs with reported costs; token totals
are summed across the available outputs. Missing PDF costs have a separate count.

```bash
PYTHONPATH=src .venv/bin/python -m evaluation \
  --mcq-root outputs/json_crops_my_all \
  --experiment-name pdf_to_json_my --device cpu

PYTHONPATH=src .venv/bin/python -m evaluation \
  --mcq-root outputs/pdf_to_jsons_all \
  --experiment-name pdf_to_json --device cpu
```

Both commands append one row each to `results/experiment.csv`. Use `--years` to
choose a different set of years, or `--golden-root` for a different golden tree.
The existing `--mcq-json` option still evaluates one paper.

In combined mode, a duplicated question number is resolved only when exactly one
candidate has all four option texts and a printed answer. This handles numbered
cover-page instructions in the original 2023 and 2024 outputs. Ambiguous duplicates
still fail with an explicit error.
