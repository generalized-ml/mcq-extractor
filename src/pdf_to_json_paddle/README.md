# PaddleOCR question pipeline

This is a separate pipeline whose default output is
`outputs/json_crops_paddle/<pdf-name>/`.

1. Reuse verified images from the existing Docling crop pipeline. Without a crop
   directory, invoke that same crop implementation, including Apple Vision for
   boundary checks. Existing crop files are copied into the new run and verified
   against their recorded hashes and source PDF.
2. Load `PP-OCRv5_mobile_det` and `PP-OCRv5_mobile_rec` once and recognize each
   crop part. No Paddle layout analysis, document orientation, unwarping, or
   text-line orientation model is used. The multilingual mobile recognizer
   supports English. Keep text detection enabled to locate lines within crops.
   Use `--ocr-backend vl` to select the previous PaddleOCR-VL-1.6 backend.
3. Parse the text into question, four options, printed answer, and printed
   solution. This stage uses local parsing rules, preserves LaTeX, normalizes
   option labels to 1–4, and never solves questions.
4. Review only questions selected by the confidence rules below. The OpenRouter
   request contains the draft JSON and complete OCR text. It contains no images.
   High-confidence questions are copied directly into the final JSON.

## Install and run

From the repository root:

```bash
source .venv/bin/activate
python -m pip install -e '.[paddle-mobile,eval]'
bash json_crops_paddle.sh
```

The shell script defaults to 2021 and reuses `outputs/json_crops_fast/2021` when
present. It loads `OPENROUTER_API_KEY` from the root's ignored `.env` file.
`OPENROUTER_MODEL` or `--review-model` chooses the remote text reviewer.
The default is `google/gemini-2.5-flash-lite`.

For another PDF or an explicit crop run:

```bash
python -m pdf_to_json_paddle dataset/pdfs/2021.pdf \
  --crops-dir outputs/json_crops_fast/2021 \
  --output outputs/json_crops_paddle
```

If no crops exist, install `.[crop,paddle-mobile,eval]` and omit `--crops-dir`; optionally
pass `--raw-json dataset/jsons/2021/raw/2021_raw.json` to reuse Docling conversion.
The existing crop engine currently requires macOS Vision access. Reusing its
saved crops does not require macOS OCR, and supports execution on another host.

First run downloads mobile model weights. Mobile `--device auto` uses CPU;
`--device cuda` requires a compatible GPU PaddlePaddle installation. Mobile does
not support MPS or `--local-files-only`. For upright PDF crops orientation
classification is unnecessary. This uses the PaddleOCR 3.x API rather than legacy
2.x `det_model_dir`/`rec_model_dir` parameters.
[Official mobile configuration](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/OCR.html).

Existing VL checkpoints cannot be resumed with the changed backend/code. Start
in a new directory to preserve the previous run:

```bash
JSON_CROPS_PADDLE_OUTPUT=outputs/json_crops_paddle_mobile \
  bash json_crops_paddle.sh --ocr-only --limit 2
```

For VL, install `.[paddle]` and pass `--ocr-backend vl`. Its `auto` device chooses
CUDA or CPU; explicit MPS is experimental. VL-only options are `--ocr-model`,
`--revision`, `--max-tokens`, and `--local-files-only`.

## Bounded runs, resume, and evaluation

```bash
# Local OCR and JSON only; inspect two questions before enabling remote review.
bash json_crops_paddle.sh --ocr-only --limit 2

# Continue OCR and review only selected questions.
bash json_crops_paddle.sh --resume

# Evaluate when all OCR and selected reviews are complete.
bash json_crops_paddle.sh --resume \
  --golden-json dataset/golden/2021_mcq_dataset/2021_mcqs.json
```

`--limit` bounds new OCR questions. `--max-llm-questions` independently bounds new
question reviews; `--retries` bounds retries per review. Missing API credentials
do not prevent local OCR: results are saved, and the command explains how to
resume if any questions require review. `--ocr-only` explicitly defers all calls.
Successful reviews and completed transcriptions are reused on resume. Failed
reviews can be retried without repeating OCR. Input, crop, code, and settings
changes require a fresh output directory. The resolved model revision is saved
and reused; `--revision COMMIT` can pin it before the first run.

Set `JSON_CROPS_PADDLE_OUTPUT` to choose another output base in the shell script.
For VL, `--local-files-only` prevents model downloads. Evaluation uses the cached local
sentence-transformer model and writes to `results/experiment.csv`.
If its weights are not cached, run the standalone evaluator without
`--local-files-only` once to download them. Partial source crop runs report their
missing crop count; evaluation is deferred until all planned work is complete.

## Confidence and review rules

Mobile confidence stops at the first whole-word `Solution` or `Answer` marker
(case-insensitive) plus three following words, across crop continuations. With
no marker, all text is scored. The boundary line retains its original line score;
Paddle does not supply individual word scores. Full OCR text remains available
for extraction. `confidence_scope` records the included text and excluded parts.
This scope controls confidence-based LLM routing; structural checks still apply.

The mobile adapter aggregates recognition-line confidence scores using their
geometric mean and fraction below 0.5. The shared `low_token_fraction` field
counts lines for mobile; `confidence_kind` records this distinction. Low-confidence
lines are retained for review. These scores do not detect every omitted formula
or misread symbol; mobile OCR is not a formula-to-LaTeX model.

The VL adapter measures the geometric mean of generated token probabilities,
excluding special tokens, and the fraction of tokens whose probability is below
0.5. It stores only selected-token probabilities during decoding rather than all
vocabulary logits. These are **uncalibrated model likelihoods**, not probabilities
that a transcription is correct.

Default routing sends a question to the text reviewer if any of the following hold:

- Any crop score is below `--confidence-threshold 0.85`.
- Any crop's uncertain-token fraction exceeds `--max-low-token-fraction 0.10`.
- Confidence is missing, transcription is empty, or generation hits the token limit.
- Parsing finds missing/duplicate options, a missing stem, or missing printed answer.
- Crop boundaries were limited by the existing crop engine's safety limits.

Use `--allow-missing-answer` for papers without printed answers. A present printed
explanation is copied; the pipeline never derives an answer by solving it.
Tune the thresholds against labeled question crops before treating them as a
quality gate. Each question saves the exact reasons in `routing.reasons`.

Text-only review can repair grouping and JSON structure. It cannot verify a
misread sign, exponent, or diagram against the image. Such OCR uncertainty remains
flagged after review. The LLM's `complete` status is not human verification.

## Artifacts

```text
outputs/json_crops_paddle/2021/
├── pipeline.json                  # Identity, crop plan, checkpoints, model revision
├── mcqs.json                      # Local PaddleOCR → structured JSON
├── final_mcqs.json                # Local high-confidence + reviewed low-confidence MCQs
└── questions/0000_q1/
    ├── patch.png                  # Copied Docling crop
    ├── part_01.png                # Source parts, preserved in reading order
    ├── paddle_ocr.json            # Per-part transcription, scores, timings, model info
    ├── paddle_ocr.txt             # Combined source text
    └── mcq.json                   # Local structured draft and routing decision
```

Every remote attempt is recorded in `llm_calls`, including reported usage/cost and
failed responses. The existing evaluator reads that ledger automatically.
Inference that runs locally has no remote API charge; local compute is not priced.
