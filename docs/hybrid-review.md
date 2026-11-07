# Hybrid MCQ extraction

Run the existing PDF converter and heuristic extractor first. This tool reads
their two JSON files; it does not rerun PDF conversion. Python 3.10+ is enough:
there are no additional Python dependencies.

Only `questions` with `needs_review: true` (also accepts `need_review`) and all
`review_candidates` are sent to OpenRouter. Other questions retain their data.

Local formula enrichment in `run_converter.py` is disabled. The vision prompt
requires all formulas in the stem/options to be preserved as LaTeX, including
exponents, subscripts, fractions and roots, using page images to correct flattened
OCR text. Unreadable math must remain flagged instead of being guessed.

An unflagged question can still contain a flattened exponent. Pass `--review-all`
to send every question and candidate for vision review. This increases API usage.
Formula accuracy still needs source comparison; schema validation cannot prove it.

## Run

From the workspace root, set your key without putting it into a script:

```bash
read -s 'OPENROUTER_API_KEY?OpenRouter API key: '
export OPENROUTER_API_KEY
export OPENROUTER_MODEL='your-provider/your-vision-model'
```

Choose an OpenRouter model/provider that supports **image input and structured
outputs**. The model above is a placeholder, not an actual model ID.

Preview evidence selection without a key, model download, or API call:

```bash
python -m experiments.docling.review \
  tests/fixtures/converter_out/2021_conv_res.json \
  tests/fixtures/converter_out/2021_conv_res_mcqs.json \
  --dry-run
```

Try one question, then resume to process the rest:

```bash
python -m experiments.docling.review \
  tests/fixtures/converter_out/2021_conv_res.json \
  tests/fixtures/converter_out/2021_conv_res_mcqs.json \
  --limit 1

python -m experiments.docling.review \
  tests/fixtures/converter_out/2021_conv_res.json \
  tests/fixtures/converter_out/2021_conv_res_mcqs.json \
  --resume
```

You can instead pass the `<name>_raw.json` and `<name>_mcqs.json` produced by
`python -m experiments.docling`. Paths are positional; filenames are not hardcoded.
Use `--output path.json` to choose the destination. The default appends
`_hybrid.json` to the MCQ input stem. Dry runs write a separate `_plan.json`.

## Evidence and output

Each request includes the candidate JSON, referenced source elements (with
coordinates), table cells, picture child text, and embedded images. Full page
images come first, followed by referenced picture/table images. Pages help
recover formulas and option labels that were absent from OCR text. Image labels
are document references, allowing graphical options to cite `image_refs`.

Only embedded `data:image/...;base64,...` images are supported. External files
or URLs are not fetched. The default cap is eight images per question; use
`--max-images` to change it. Missing or omitted evidence is recorded as warnings.
Only the pages already associated with the candidate are supplied. If conversion
omitted a continuation page or its image, it cannot be reconstructed here.

The model copies printed answers; it is instructed not to solve questions.
Unknown answers stay null. The response is validated for question identity,
option labels, answer membership, image references, and required fields.
Validation checks structure, not factual accuracy. A complete model response
with no warnings clears `needs_review`; it is not human verification.

Recovered candidates enter `questions`. Incomplete candidates remain in
`review_candidates`. Original source references/pages are preserved; original
heuristic warnings are stored as `heuristic_warnings`. Failed calls retain the
original candidate with `llm_error` and `needs_review: true`.

Output is checkpointed after every question. `--resume` reuses successful
responses only when input hashes, model and image cap match. Failed requests
are retried on resume. Successful responses that still need review are cached
too; start a fresh output to reprocess them. `--limit` counts new questions,
not HTTP retries. Transient HTTP/network failures and invalid responses retry
twice by default; permanent API errors retain the candidate for review.
The formula-preserving prompt uses a new checkpoint version. For an older hybrid
run, choose a new `--output` path and omit `--resume` so old responses are not reused.

Selected source content and images are sent to OpenRouter and its model
provider. Requests can incur charges. No API key is written to output files.

API references:
- https://openrouter.ai/docs/guides/features/structured-outputs
- https://openrouter.ai/docs/guides/overview/multimodal/image-understanding

## Offline tests

```bash
PYTHONPATH=src:. python -m unittest discover -s tests -p 'test_hybrid.py'
```
