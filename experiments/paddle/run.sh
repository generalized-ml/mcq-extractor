#!/usr/bin/env bash
# 2021: reuse Docling crops → PaddleOCR → structured JSON → selective LLM review.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${DOCLING_PYTHON:-$REPO_ROOT/.venv/bin/python}"
ARGS=()
if [[ -f "$REPO_ROOT/outputs/json_crops_fast/2021/crop_pipeline.json" ]]; then
    ARGS+=(--crops-dir "$REPO_ROOT/outputs/json_crops_fast/2021")
elif [[ -f "$REPO_ROOT/dataset/jsons/2021/raw/2021_raw.json" ]]; then
    ARGS+=(--raw-json "$REPO_ROOT/dataset/jsons/2021/raw/2021_raw.json")
fi
exec "$PYTHON" -u -m experiments.paddle \
    "$REPO_ROOT/dataset/pdfs/2021.pdf" \
    --output "${JSON_CROPS_PADDLE_OUTPUT:-$REPO_ROOT/outputs/json_crops_paddle}" \
    "${ARGS[@]}" "$@"
