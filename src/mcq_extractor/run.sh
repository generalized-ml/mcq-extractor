#!/usr/bin/env bash
# Any year: reuse Docling crops → HF mobile OCR → structured JSON → selective LLM review.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR/src:$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${DOCLING_PYTHON:-$SCRIPT_DIR/.venv/bin/python}"
PDF="${1:-$SCRIPT_DIR/dataset/pdfs/2021.pdf}"
if [[ $# -gt 0 && "$1" == *.pdf ]]; then
    shift
else
    PDF="$SCRIPT_DIR/dataset/pdfs/2021.pdf"
fi
STEM="$(basename "$PDF" .pdf)"
ARGS=()
if [[ -f "$SCRIPT_DIR/outputs/json_crops_fast/$STEM/crop_pipeline.json" ]]; then
    ARGS+=(--crops-dir "$SCRIPT_DIR/outputs/json_crops_fast/$STEM")
elif [[ -f "$SCRIPT_DIR/dataset/jsons/$STEM/raw/${STEM}_raw.json" ]]; then
    ARGS+=(--raw-json "$SCRIPT_DIR/dataset/jsons/$STEM/raw/${STEM}_raw.json")
fi
exec "$PYTHON" -u -m mcq_extractor \
    "$PDF" \
    --output "${MCQ_OUTPUT:-$SCRIPT_DIR/outputs/native_ocr}" \
    "${ARGS[@]}" "$@"
