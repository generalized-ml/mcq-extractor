#!/usr/bin/env bash
# Any year: reuse Docling crops → HF mobile OCR → structured JSON → selective LLM review.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${DOCLING_PYTHON:-$REPO_ROOT/.venv/bin/python}"
PDF="${1:-$REPO_ROOT/dataset/pdfs/2021.pdf}"
if [[ $# -gt 0 && "$1" == *.pdf ]]; then
    shift
else
    PDF="$REPO_ROOT/dataset/pdfs/2021.pdf"
fi
STEM="$(basename "$PDF" .pdf)"
ARGS=()
if [[ -f "$REPO_ROOT/outputs/json_crops_fast/$STEM/crop_pipeline.json" ]]; then
    ARGS+=(--crops-dir "$REPO_ROOT/outputs/json_crops_fast/$STEM")
elif [[ -f "$REPO_ROOT/dataset/jsons/$STEM/raw/${STEM}_raw.json" ]]; then
    ARGS+=(--raw-json "$REPO_ROOT/dataset/jsons/$STEM/raw/${STEM}_raw.json")
fi
exec "$PYTHON" -u -m mcq_extractor \
    "$PDF" \
    --output "${MCQ_OUTPUT:-$REPO_ROOT/outputs/native_ocr}" \
    "${ARGS[@]}" "$@"
