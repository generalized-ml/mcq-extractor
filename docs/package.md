# MCQ Extractor

Extract structured multiple-choice questions from question-paper PDFs using
local PP-OCRv5 models, Docling crops, and optional selective LLM review.

## Install

Requires Python 3.10 or newer on macOS or Linux:

```bash
python -m pip install mcq-extractor
# Optional evaluation and plots:
python -m pip install 'mcq-extractor[eval]'
```

## Extract a PDF

```bash
mcq-extract paper.pdf --output outputs/my-paper --ocr-only
```

macOS uses Apple Vision for crop checks. To run question OCR on Apple Silicon:

```bash
mcq-extract paper.pdf --device mps --workers 1 --ocr-only
```

On Linux, crop generation uses native PP-OCRv5. CUDA is selected when available:

```bash
mcq-extract paper.pdf --crop-ocr hf --device auto --ocr-only
```

Models download on first use and are cached. For selected text-only LLM review,
set `OPENROUTER_API_KEY` in your environment or a working-directory `.env`, then
omit `--ocr-only`. Remote review can incur API charges. Optional
`OPENROUTER_MODEL` selects the review model.

Results include `mcqs.json`, `final_mcqs.json`, `pipeline.json`, `runtime.json`,
and source question crops under the output directory. Use a fresh directory
when changing settings; use `--resume` for interrupted runs with matching inputs.
Installed commands use the working directory for default outputs and `.env`;
set `MCQ_WORKSPACE` to override it.

## Evaluate

```bash
evaluate-mcqs outputs/my-paper /path/to/golden --no-save
```

This project is in development. Inspect uncertain OCR, formulas, and answer
labels against the source images. Model weights and datasets are not included
in this distribution and retain their own licenses.

[Source and full documentation](https://github.com/generalized-ml/mcq-extractor)
· [Issues](https://github.com/generalized-ml/mcq-extractor/issues)

MIT licensed. Alternative pipelines and benchmarks live in the GitHub
repository's `experiments/` directory and are not installed by this package.
