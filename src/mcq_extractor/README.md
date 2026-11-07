# PDF → MCQ JSON: native Hugging Face mobile OCR

This package owns the OCR, confidence, parsing, review orchestration, checkpoints,
and labeling export. It reuses the repository's proven `pdf_to_jsons` Docling
crop engine, OpenRouter transport/schema, and evaluator. It does not import or
require `paddleocr` or `paddlepaddle`.

See the [project overview](../../README.md) and editable
[draw.io pipeline design](../../docs/low-cost-mcq-extractor.drawio) for the complete
flow and local OCR model architecture.

## Setup and run

```bash
.venv/bin/python -m pip install -e '.[ocr]'
bash src/mcq_extractor/run.sh --ocr-only --limit 2
bash src/mcq_extractor/run.sh --resume
```

The shell script selects the 2021 PDF and existing `outputs/json_crops_fast/2021`
crops. Outputs default to `outputs/json_crops_my/2021`. Use
`MCQ_OUTPUT=outputs/my_experiment` to keep separate experiments.
An existing output from changed code/settings requires a new output directory.

```bash
PYTHONPATH=src:. .venv/bin/python -m mcq_extractor dataset/pdfs/2021.pdf \
  --crops-dir outputs/json_crops_fast/2021 --output outputs/my_experiment
```

Without `--crops-dir`, the shared Docling crop stage runs; install `.[ocr,crop]`.
`--crop-ocr auto` selects Apple Vision on macOS and native PP-OCRv5 on Linux or
Windows. Force either with `--crop-ocr vision` / `--crop-ocr hf`. The native crop
adapter reads PIL images in memory and returns text, confidence, and pixel boxes
to the same crop-boundary protection and shrink/fallback logic. Confidence and
recognition quality can differ from Apple Vision; compare the saved crop images.
For the portable path, initial Docling PDF conversion uses RapidOCR with its
PyTorch backend, so it also avoids Apple Vision and does not require ONNX Runtime.
`--crop-device auto|cpu|cuda|mps` controls crop-check inference separately from
the main OCR stage's `--device`. Native crop checks use the default pinned mobile
models; custom `--det-model` / `--rec-model` apply to the main OCR stage.
`--local-files-only` applies to native crop and question OCR weights; Docling's
separate layout/table/RapidOCR model assets may still download on first conversion.
No layout model runs in the crop OCR stage.
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

## Parallel questions across a PDF

CPU OCR automatically detects the number of **physical** CPU cores, `N`, and
starts `max(1, N // 2)` persistent question workers. Each uses two PyTorch CPU
threads (one on a single-core machine) and loads the detector/recognizer once.
On an eight-core CPU this gives four concurrent questions; on a sixteen-core
CPU it gives eight. The number of workers is capped at the number of pending
questions after applying `--limit` and excluding completed checkpoints.

For `X` pending questions and `P` workers, each handles approximately
`ceil(X / P)` questions. A shared task queue assigns complete questions to free
workers instead of waiting for a fixed group to finish. All continuation crops
for one question stay together and retain their original order. Recognition
inside each worker retains batches of eight; there is no nested process pool.
The two CPU threads execute a question's model operations; individual threads
are not pinned to dedicated physical cores.

```bash
# Use a fresh output directory when migrating from the earlier code version.
MCQ_OUTPUT=outputs/json_crops_my_parallel \
  bash src/mcq_extractor/run.sh --ocr-only

# Limit memory use or run serially.
bash src/mcq_extractor/run.sh --workers 2 --ocr-only
bash src/mcq_extractor/run.sh --workers 1 --ocr-only
```

`--physical-cores N` overrides unavailable core detection. GPU inference uses
one worker; multiple question workers require CPU. Models are resolved/downloaded
once in the parent before starting workers, which then load local weights only.
Each worker has its own models and inference memory; reduce `--workers` if needed.

Only the parent writes OCR artifacts and checkpoints as questions complete.
MCQ output and LLM review retain plan order even if OCR finishes out of order.
On resume, completed questions are skipped; worker settings can change without
invalidating results. `pipeline.json` records the allocation, and each question
records its worker PID, OCR time, and model initialization time. All OCR workers
exit before selective LLM review and evaluation begin. The Docling crop stage
and remote review retain their existing scheduling.

## Timing each OCR stage

Profile a saved question crop with cached models:

```bash
.venv/bin/python -m mcq_extractor.profiling \
  outputs/json_crops_my_all/2021/questions/0002_q3/patch.png --precision float32
```

Use `--precision float16` for the local copies created by `model_quantization`.
Each command measures model initialization separately, then a first and warm
OCR pass. Reports default to `outputs/ocr_timing/stages_<precision>.json`;
`--output` selects another path. `--device` and `--batch-size` select execution
settings. Only cached/local models are loaded.

Timings include preprocessing, detector backbone/feature fusion/head, polygon
extraction, line cropping, recognizer backbone/SVTR encoder/classifier, and CTC
decoding. Recognition times sum all line batches. The SVTR encoder includes
convolutions around its attention blocks. Other model operations include pooling,
softmax, and framework overhead. Stage rows exclude nested totals to avoid double
counting. GPU work is synchronized; profiling adds overhead, so compare profiled
runs with each other. Normal `reader.read(path)` does not attach profiling hooks;
`reader.read(path, profile=True)` adds a `timings` field to its result.

## Benchmarking parallel recognition

Compare one process using four PyTorch threads with two persistent recognition
workers using two threads each, keeping FP32 and batches of eight:

```bash
.venv/bin/python -m experiments.recognition.parallel_benchmark \
  outputs/json_crops_my_all/2021/questions/0002_q3/patch.png
```

The detector runs in the parent process. Recognition batches run in workers
started with `spawn`; each loads its recognizer once. Image arrays are passed
between processes and results are restored in batch/reading order. No LLM calls
or pipeline checkpoint changes are made. All models must already be cached.

The report at `outputs/ocr_timing/multiprocessing.json` includes initialization,
three passes for each configuration (first plus two warm repeats), warm median
speedup, and exact text/box comparisons against serial output. Recognition wall
time includes dispatch, data transfer, and waiting for workers. Worker CPU times
overlap and must not be added to wall time. Use `--workers`,
`--threads-per-worker`, `--serial-threads`, `--batch-size`, `--runs`, or `--output`
to vary the experiment. This is a benchmark; the standard pipeline continues to
use its existing reader.

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
# Install .[ocr,eval] for evaluation dependencies.
bash src/mcq_extractor/run.sh --resume \
  --golden-json dataset/golden/neet_2021/2021_mcqs.json
```

Evaluation waits until crops, OCR, and selected reviews are complete. It uses the
cached sentence-transformer model and writes `results/experiment.csv`.

## Preparing for fine-tuning

```bash
PYTHONPATH=src:. .venv/bin/python -m mcq_extractor.export_labels \
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
bash src/mcq_extractor/run.sh --rec-model /absolute/path/to/finetuned_rec \
  --output outputs/my_finetuned --ocr-only --limit 2
```

Use `--det-model` for a custom detector. For custom Hugging Face repositories,
provide the matching `--rec-revision`/`--det-revision` commit explicitly (the
built-in defaults pin official repos). Local model JSON and safetensors hashes
are included in checkpoint identity to prevent mixing weights across resumes.
The processor's vocabulary must match the recognition head.

References: [native recognition API](https://huggingface.co/docs/transformers/en/model_doc/pp_ocrv5_mobile_rec),
[official detector weights](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_det_safetensors).
