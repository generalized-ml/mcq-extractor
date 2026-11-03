# PDF to JSON and MCQs
## This repo is currenting in work-in-progress mode - 
![alt text](docs/result/experiments.png)
For the separate PaddleOCR-VL-1.6 pipeline with confidence-based text-only LLM
review, see [Paddle pipeline setup and usage](src/pdf_to_json_paddle/README.md).

Convert NEET question-paper PDFs into Docling source JSON, heuristic MCQs, and
MCQs reviewed with OpenRouter vision. The pipeline and evaluator were copied
from `/Users/raghavsharma/Documents/My/Projects/NEET/codebase` and organized as
the `pdf_to_jsons` package. The existing local Docling checkout is retained.

```text
.
├── src/pdf_to_jsons/          # Complete conversion pipeline and CLI tools
│   ├── run_converter.py      # PDF → Docling ConversionResult JSON
│   ├── extract_mcqs.py       # Source JSON → structured MCQs
│   ├── pdf_to_mcqs.py        # Conversion, extraction, review, checkpoints
│   ├── hybrid_mcq/run.py     # OpenRouter vision review
│   ├── evaluate.py           # Compare predictions with golden questions
│   └── show_mcq.py           # Inspect a question and its source images
├── dataset/
│   ├── pdfs/                # Source PDFs, 2021–2025
│   ├── golden/              # Reference questions and image assets
│   └── jsons/               # Existing conversion outputs copied from NEET
├── docling/                 # Retained local Docling repository
├── tests/                   # Offline regression tests
│   └── fixtures/            # Preserved data from the former testing sandbox
├── docs/                    # Architecture, evaluation, and hybrid review
├── convert_pdfs_to_jsons.sh  # Batch conversion, all PDFs
├── pyproject.toml           # Package metadata and command entry points
└── requirements.txt         # Installs local Docling and this package
```

## Setup

From this workspace root, use Python 3.10 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The local Docling source is preferred when running from this checkout. Docling
conversion may download model weights on first use. Heuristic extraction and
hybrid review use Python's standard library; evaluation has separate dependencies.

## Convert one PDF

Generate source JSON and heuristic MCQs, plus an LLM evidence plan, without API calls:

```bash
python -m pdf_to_jsons dataset/pdfs/2021.pdf outputs/jsons --dry-run
```

To run vision review, export `OPENROUTER_API_KEY` in your shell, then resume:

```bash
python -m pdf_to_jsons dataset/pdfs/2021.pdf outputs/jsons --resume
```

Selected question evidence and embedded images are sent to OpenRouter. Requests
can incur charges. Default model: `google/gemini-2.5-flash-lite`; change it using
`OPENROUTER_MODEL` or `--model`. Review normally covers flagged questions and
incomplete candidates. Use `--review-all` to include every question, or `--limit 5`
for a small initial run. Uncertain results remain flagged for review.

```text
outputs/jsons/2021/
├── pipeline.json
├── raw/2021_raw.json
├── mcqs/2021_mcqs.json
└── mcq_with_llm/2021_mcqs_with_llm.json
```

`--dry-run` writes an evidence plan in the last folder. `--resume` checks PDF and
code hashes and reuses completed stages. Existing output is protected from
overwriting. Copied `dataset/jsons` files retain their original paths and hashes
as provenance; use `outputs/` for new runs. Those old checkpoints cannot be resumed
after moving the inputs and code. Inspection of copied outputs may require
`--raw-json` to point to the relocated raw JSON explicitly.

## Convert all PDFs

The batch script processes every PDF, including 2021. It defaults to
`dataset/pdfs` and `dataset/jsons`; specify a new output directory to keep the
copied results intact:

```bash
./convert_pdfs_to_jsons.sh dataset/pdfs outputs/jsons --dry-run
./convert_pdfs_to_jsons.sh dataset/pdfs outputs/jsons --resume
```

Set `DOCLING_PYTHON` to override the Python executable. Otherwise the script
prefers this workspace's `.venv/bin/python`, then `python3`.

## Individual tools

```bash
python -m pdf_to_jsons.run_converter --pdf dataset/pdfs/2021.pdf --output-dir outputs/raw
python -m pdf_to_jsons.extract_mcqs --help
python -m pdf_to_jsons.hybrid_mcq.run --help
python -m pdf_to_jsons.show_mcq outputs/jsons/2021/mcq_with_llm/2021_mcqs_with_llm.json 40
```

After installation, `pdf-to-jsons`, `pdf-to-raw-json`, `extract-mcqs`,
`review-mcqs`, `evaluate-mcqs`, and `show-mcq` provide the same commands.

## Evaluation and tests

```bash
python -m pip install -e '.[eval]'
python -m pdf_to_jsons.evaluate \
  --mcq-json dataset/jsons/2021/mcq_with_llm/2021_mcqs_with_llm.json \
  --golden-json dataset/golden/2021_mcq_dataset/2021_mcqs.json

PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'
```

Evaluation writes to `results/pdf_to_jsons`. Offline tests need no model downloads
or API key. Evaluation also saves the recorded LLM extraction costs in USD and
token counts in the experiment summary, metadata, and a per-call CSV. Missing
costs stay unknown; unrecorded failed/retried attempts are excluded.
See [crop-based OCR and LLM extraction](docs/crop-pipeline.md),
[architecture](docs/architecture.md), [evaluation](docs/evaluation.md),
and [hybrid review](docs/hybrid-review.md) for details.

For the PaddleOCR-free mobile pipeline using native Hugging Face weights, see
[src/pdf_to_json_my/README.md](src/pdf_to_json_my/README.md). Run
`bash json_crops_my.sh`; install its dependencies with `pip install -e '.[my]'`.
