# Crop detection and extraction

`json_crops.sh` processes 2021 using existing Docling JSON when available, crops
question regions, runs local OCR, sends the finalized images to OpenRouter,
and evaluates final MCQs against the golden dataset. It loads the API key from
the ignored `.env` file. The default is Apple Vision on macOS and native
PP-OCRv5 on Linux/Windows. The portable path requires `.[my,crop]`.

On macOS with Vision access:

```bash
python -m pip install -e '.[crop,eval]'
JSON_CROPS_OUTPUT=outputs/json_crops_fast bash json_crops.sh
```

On Linux, install `.[my,crop,eval]` and select `--ocr-backend hf` for the shared
crop CLI. The native end-to-end pipeline exposes the same choice as `--crop-ocr hf`.
Portable crop checks return pixel boxes and line confidence through the existing
OCR contract. Initial Docling conversion uses RapidOCR with `backend='torch'`;
macOS Vision conversion remains available through the Vision crop backend.
Docling's official [OCR installation guide](https://docling-project.github.io/docling/getting_started/installation/)
lists its supported OCR engines. OCR output is not guaranteed to match Vision.

## Reverse crop detection

Each question begins at its Docling anchor and initially ends immediately before
the next question anchor. For two-column pages or page continuations this is a
sequence of column/page rectangles in reading order. The next question is never
included in the crop. The final question uses the remaining layout regions,
bounded by `--max-regions`.

The detector first runs OCR on the full bounded patch. It then moves the bottom
edge upward in `--step` increments (default 24 PDF points), removing only strips
below both the OCR text boxes and Docling's layout content. Diagrams, formulas,
tables, and explanation lines remain protected even when OCR cannot read them.
Empty-strip checks use geometry and do not make another OCR call per increment.
Each region in a continuation keeps its own width and page coordinate system.

A separate accurate OCR pass reads the finalized patch. If its text differs
from the full patch after whitespace normalization, the detector restores the
full crop and reruns OCR. Normally this takes two question OCR calls; a fallback
takes three. Apple Vision also runs a small preflight once per invocation.
Structural failures remain flagged for the following crop-only LLM stage.

Saved `growth.json` files retain their filename for compatibility, but now record
`strategy: next_anchor_shrink` and the full-boundary, final-crop, and optional
fallback OCR checks.

## Existing runs

The new algorithm changes the code identity. Use a fresh output directory for
outputs created with the earlier growing-crop algorithm; they cannot be resumed
with this version. Existing crop images and LLM results remain on disk.
To resume a run created with this version:

```bash
JSON_CROPS_OUTPUT=outputs/json_crops_fast bash json_crops.sh --resume
```

Run just cropping and OCR without LLM calls or evaluation:

```bash
JSON_CROPS_OUTPUT=outputs/json_crops_fast bash json_crops.sh --detect-only --limit 3
```

Results are saved under `<output>/2021/`; final extraction is `final_mcqs.json`.
Evaluation defaults to `results/json_crops/2021/`. Region detection remains a
layout heuristic; inspect flagged crops and validate numerical/math transcription.
