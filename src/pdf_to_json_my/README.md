# PDF → MCQ JSON: native Hugging Face mobile OCR

This package owns the OCR, confidence, parsing, review orchestration, checkpoints,
and labeling export. It reuses the repository's proven `pdf_to_jsons` Docling
crop engine, OpenRouter transport/schema, and evaluator. It does not import or
require `paddleocr` or `paddlepaddle`.

## Setup and run

```bash
.venv/bin/python -m pip install -e '.[my]'
bash json_crops_my.sh --ocr-only --limit 2
bash json_crops_my.sh --resume
```

The shell script selects the 2021 PDF and existing `outputs/json_crops_fast/2021`
crops. Outputs default to `outputs/json_crops_my/2021`. Use
`JSON_CROPS_MY_OUTPUT=outputs/my_experiment` to keep separate experiments.
An existing output from changed code/settings requires a new output directory.

```bash
PYTHONPATH=src .venv/bin/python -m pdf_to_json_my dataset/pdfs/2021.pdf \
  --crops-dir outputs/json_crops_fast/2021 --output outputs/my_experiment
```

Without `--crops-dir`, the existing Docling crop stage runs; install `.[my,crop]`
for Apple Vision boundary checks. No layout model runs in the crop OCR stage.
Text-line detection remains necessary to isolate lines within each MCQ image.

## Models and caching

The defaults are the official native Transformers conversions:

- `PaddlePaddle/PP-OCRv5_mobile_det_safetensors`
- `PaddlePaddle/PP-OCRv5_mobile_rec_safetensors`

Both are pinned to specific Hugging Face commits in `ocr.py`. Local directories
and the Hugging Face cache are checked first; missing JSON/weight files download
on first use. Downloaded repository Python code is not executed. PyTorch,
Transformers, OpenCV and their dependencies are required; model weights alone
are not a runnable inference engine. `--local-files-only` prevents downloads.

Recognition runs in batches of 8 (`--batch-size`), preserving each line's aspect
ratio before padding. `--device auto` selects CUDA or CPU. Explicit `--device mps`
is experimental and fails clearly if unavailable. There is no autoregressive VL
model, orientation classifier, unwarping, or document layout model here.

## Extraction and review

Full text, line polygons, scores and timings are stored in each question's
`paddle_ocr.json`. Confidence stops at the first whole-word `Solution` or `Answer`
plus three words, across crop continuations. The boundary line contributes its
whole score; without a marker all lines count. Full OCR text is retained.

Local rules parse the question, four options, printed answer and solution. Only
low-confidence or structurally incomplete questions go to the text-only LLM.
`routing.reasons` explains each decision. `.env` supplies `OPENROUTER_API_KEY` and
optionally `OPENROUTER_MODEL`. `--ocr-only` disables all remote calls;
`--max-llm-questions` caps reviews. Every remote attempt records returned cost and
usage in `llm_calls`. Text-only review cannot reliably repair misread formulas.

Completed work is checkpointed. `mcqs.json` contains local drafts and
`final_mcqs.json` contains selected reviewed results. The previous Paddle pipeline
remains available unchanged. Native inference is a separate implementation;
compare against golden data before assuming identical recognition quality.

```bash
# Install .[my,eval] for evaluation dependencies.
bash json_crops_my.sh --resume \
  --golden-json dataset/golden/neet_2021/2021_mcqs.json
```

Evaluation waits until crops, OCR, and selected reviews are complete. It uses the
cached sentence-transformer model and writes `results/experiment.csv`.

## Preparing for fine-tuning

```bash
PYTHONPATH=src .venv/bin/python -m pdf_to_json_my.export_labels \
  outputs/json_crops_my/2021 dataset/ocr_labels/2021
```

This exports rectified line PNGs and `labels.jsonl` with `suggested_text`, empty
`text`, `verified: false`, polygons, question IDs and source PDF hashes. Human
reviewers must fill exact transcriptions into `text` and mark `verified: true`.
OCR suggestions and LLM rewrites are not ground truth. The exporter refuses to
overwrite existing corrections. Split train/validation/test by PDF (or exam),
not randomly by line, to avoid leaking the same question across splits.

Recognition training needs line images and exact text; detector training needs
corrected polygons, including missed lines. MCQ JSON alone is insufficient for
both. Training is a future step; no training job or automatic labels are created
by this pipeline. Save a future Transformers-compatible model and its processor
with `save_pretrained()` and select it without changing orchestration:

```bash
bash json_crops_my.sh --rec-model /absolute/path/to/finetuned_rec \
  --output outputs/my_finetuned --ocr-only --limit 2
```

Use `--det-model` for a custom detector. For custom Hugging Face repositories,
provide the matching `--rec-revision`/`--det-revision` commit explicitly (the
built-in defaults pin official repos). Local model JSON and safetensors hashes
are included in checkpoint identity to prevent mixing weights across resumes.
The processor's vocabulary must match the recognition head.

References: [native recognition API](https://huggingface.co/docs/transformers/en/model_doc/pp_ocrv5_mobile_rec),
[official detector weights](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_det_safetensors).
