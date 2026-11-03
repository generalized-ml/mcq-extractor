#!/usr/bin/env bash
# 2021: reuse Docling crops → PaddleOCR → structured JSON → selective LLM review.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${DOCLING_PYTHON:-$SCRIPT_DIR/.venv/bin/python}"
ARGS=()
if [[ -f "$SCRIPT_DIR/outputs/json_crops_fast/2021/crop_pipeline.json" ]]; then
    ARGS+=(--crops-dir "$SCRIPT_DIR/outputs/json_crops_fast/2021")
elif [[ -f "$SCRIPT_DIR/dataset/jsons/2021/raw/2021_raw.json" ]]; then
    ARGS+=(--raw-json "$SCRIPT_DIR/dataset/jsons/2021/raw/2021_raw.json")
fi
exec "$PYTHON" -u -m pdf_to_json_paddle \
    "$SCRIPT_DIR/dataset/pdfs/2021.pdf" \
    --output "${JSON_CROPS_PADDLE_OUTPUT:-$SCRIPT_DIR/outputs/json_crops_paddle}" \
    "${ARGS[@]}" "$@"
