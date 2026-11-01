# PDF-to-MCQ evaluation

Compare extracted MCQs against a golden JSON using exact word-prefix matching
and local sentence embeddings. No LLM API calls are made.

Run from the workspace root, using the Python environment where you installed the dependencies:

```bash
python -m pip install -e ".[eval]"

python -m pdf_to_jsons.evaluate \
  --mcq-json dataset/jsons/2021/mcq_with_llm/2021_mcqs_with_llm.json \
  --golden-json dataset/golden/2021_mcq_dataset/2021_mcqs.json
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
| Question | 0.1 |
| Option 1 | 0.2 |
| Option 2 | 0.2 |
| Option 3 | 0.2 |
| Option 4 | 0.2 |
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

These metrics evaluate **text**, not visual correctness. Image-only fields with
no text score zero; image paths/references are not compared. A correct answer
label is included in semantic scoring as requested, but embedding similarity of
short labels is not a measure of answer correctness. Use its keyword field score
to check exact label agreement. Likewise, semantic similarity can miss changed
numbers, negation, or mathematical symbols; keep both metrics visible.

## Output

Results default to `results/pdf_to_jsons`, independent of working directory:

- `experiments.csv`: one row per experiment with overall scores, six field scores
  for each metric, missing/extra counts, and input/model identifiers.
- `experiment_0001_questions.csv`: one row per golden question with the same field
  scores, prefix lengths, word counts, first mismatched words, and diagnostics.
- `experiment_0001_metadata.json`: scoring rules, weights, and input SHA-256 hashes.
- `experiment_0001_llm_calls.csv`: recorded extraction/review calls with response
  IDs, models, statuses, reported USD costs, and prompt/completion/total tokens.

## LLM costs

Evaluation automatically reads the extraction costs already saved in the MCQ
JSON. `experiments.csv` adds `llm_cost_usd`, `llm_cost_currency`,
`llm_recorded_calls`, `llm_priced_calls`, `llm_missing_cost_calls`, and
`llm_prompt_tokens`, `llm_completion_tokens`, and `llm_total_tokens`. The same
summary is stored under `llm_usage` in the metadata JSON and printed in the CLI.
No additional argument, API key, or LLM call is required to read costs.

Costs use OpenRouter's saved `usage.cost` value; see its
[usage accounting documentation](https://github.com/OpenRouterTeam/docs/blob/main/cookbook/administration/usage-accounting.mdx)
and [USD credit denomination](https://openrouter.ai/support).
The evaluator prefers `hybrid.results` so that usage copied into question-level
`llm` fields is not counted twice. Without that checkpoint, it reads `llm` from
both `questions` and `review_candidates`. Response IDs are deduplicated.
Costs cover the entire recorded extraction, including incomplete candidates
and questions absent from the golden set.

Missing or invalid costs remain null (blank in CSV), rather than being treated
as free. A reported zero is retained. If only some costs are present,
`llm_cost_usd` is their subtotal and `llm_missing_cost_calls` exposes the missing
records. Token totals likewise include only reported values. Historical outputs
do not retain all failed or retried HTTP attempts, so these reports cannot
reconstruct their charges or serve as a complete billing ledger. Evaluation
itself uses local embeddings and makes no LLM requests.

Existing `experiments.csv` files created before these additional columns have
a different schema. Choose a new `--output-dir` when using such an older report;
the evaluator preserves it rather than mixing incompatible rows.

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
