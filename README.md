# Low-Cost MCQ Extractor

Turn question-paper PDFs into structured MCQ JSON using local OCR and selective
text-only LLM review. The primary implementation is
[`src/pdf_to_json_my`](src/pdf_to_json_my): it extracts question text, four
options, printed answers, explanations, and source references while retaining
question crops for inspection.

**Work in progress.** Current experiments use NEET papers from 2021–2025.
Extraction quality, formula recognition, and review routing are still being
improved. Printed answers are transcribed from the supplied paper or solutions.

## Results

The existing experiment chart is retained below:

![Recorded MCQ extraction experiment results](docs/result/experiments.png)

The recorded comparison in `results/experiment.csv` contains this snapshot:

| Method | Predicted / golden questions | Keyword score | Semantic score | Aggregate score | Recorded LLM cost / PDF |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native local OCR + selective text review (`pdf_to_json_my`) | 779 / 980 | 0.5071 | 0.7200 | 0.6135 | $0.01851 |
| Original Docling + vision review (`pdf_to_json`) | 965 / 980 | 0.6877 | 0.8449 | 0.7663 | $0.04179 |

These are historical runs, not a completed comparison of both methods on all
questions. The native run had 201 missing predictions, including 199 pending
2023 questions. Cost/PDF averages PDFs with recorded costs: 4 of 5 for the native
method and 5 of 5 for the original method. There were also 47 and 53 calls,
respectively, without reported costs. Costs cover recorded LLM API usage;
local compute is excluded. This snapshot has not been recomputed for this README.

Keyword scoring uses exact matching word prefixes, so an early punctuation,
Unicode, or LaTeX difference can stop a field's score. Missing questions score
zero. See [evaluation details](docs/evaluation.md) for scoring and coverage rules.

## Pipeline design

[![Low-Cost MCQ Extractor pipeline: local OCR, selective review, and evaluation](docs/low-cost-mcq-extractor.svg)](docs/low-cost-mcq-extractor.drawio)

## Setup

Use Python 3.10 or newer from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[my,crop,eval]'
```

The `crop` extra includes macOS-specific Vision support. To process already
generated crops on another platform, install `.[my,eval]` and pass `--crops-dir`.
Model weights download on first use and are cached. `--local-files-only` requires
the native OCR models to be fully cached.

For selected LLM review, create a root `.env` with your own key:

```dotenv
OPENROUTER_API_KEY=your-key
OPENROUTER_MODEL=google/gemini-2.5-flash-lite
```

The native pipeline loads this file. Local OCR with `--ocr-only` works without
an API key.

## Extract a PDF

Start with two questions and local OCR only:

```bash
JSON_CROPS_MY_OUTPUT=outputs/my_first_run \
  bash json_crops_my.sh dataset/pdfs/2021.pdf --ocr-only --limit 2
```

Continue the remaining OCR and review selected questions:

```bash
JSON_CROPS_MY_OUTPUT=outputs/my_first_run \
  bash json_crops_my.sh dataset/pdfs/2021.pdf --resume
```

To reuse saved crops explicitly:

```bash
.venv/bin/python -m pdf_to_json_my dataset/pdfs/2021.pdf \
  --crops-dir outputs/json_crops_fast/2021 \
  --output outputs/my_crop_run --ocr-only
```

Choose a fresh output directory when PDF, OCR settings, or checkpointed code
changes. `--resume` verifies saved inputs and artifacts. The shell script defaults
to the 2021 PDF and `outputs/json_crops_my`; `JSON_CROPS_MY_OUTPUT` sets the
experiment folder. See the [native pipeline guide](src/pdf_to_json_my/README.md)
for concurrency, profiling, local models, and labeling export.

## Output format

Both methods publish the same public JSON filenames:

```text
outputs/<experiment>/<PDF-name>/
├── mcqs.json                 # Local or heuristic drafts
├── final_mcqs.json           # Final extraction and review status
├── pipeline.json             # Identity, crop plan, checkpoints, worker metadata
├── runtime.json              # Recorded pipeline wall-time sessions
└── questions/                # Native pipeline's per-question evidence
    └── <question-key>/
        ├── patch.png
        ├── part_01.png
        ├── paddle_ocr.json    # Native OCR lines, boxes, scores, and timing
        ├── paddle_ocr.txt
        └── mcq.json
```

Question records preserve source crops, page references, routing reasons, and
review flags. The original method also retains its raw Docling data and legacy
intermediate folders. `runtime.json` is available for runs made with timing
support; historical durations can remain unknown.

## Evaluate and plot

Provide an experiment folder; golden references default to `dataset/golden`:

```bash
.venv/bin/python -m evaluation outputs/json_crops_my_all
.venv/bin/python -m evaluation outputs/pdf_to_jsons_all
```

Each command discovers available PDFs and appends one row to
`results/experiment.csv`. An optional second folder selects the golden dataset.
Use `--no-save` to print results without writing a row:

```bash
.venv/bin/python -m evaluation outputs/my_first_run dataset/golden --no-save
.venv/bin/python -m evaluation outputs/json_crops_my_all --years 2023 --no-save
```

Reports include quality scores, average recorded cost/PDF, cost coverage,
available pipeline wall time, and separate evaluation duration. Folder discovery
includes only available PDFs unless `--years` explicitly requests missing years.
The first evaluation may download the embedding model; `--local-files-only`
uses cached weights.

```bash
.venv/bin/python -m evaluation.plot_experiments
# Optional vector export:
.venv/bin/python -m evaluation.plot_experiments --output results/experiments.svg
```

The chart reads `results/experiment.csv` and defaults to
`results/experiments.png`. See [evaluation documentation](docs/evaluation.md).

## Other methods and development

- [Native local OCR pipeline](src/pdf_to_json_my/README.md): primary low-cost path.
- [Paddle pipeline](src/pdf_to_json_paddle/README.md): Paddle mobile and VL backends.
- [Original pipeline architecture](docs/architecture.md): Docling conversion,
  heuristic extraction, and vision review.
- [Crop engine](docs/crop-pipeline.md) and [hybrid review](docs/hybrid-review.md).

The original method remains available through `python -m pdf_to_jsons` and
`convert_pdfs_to_jsons.sh`.

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

Tests use offline fixtures and mocked external calls. Extraction can still miss
text, confuse option labels, or alter formulas; inspect uncertain questions
against their saved images before using the JSON as verified data.
