#!/usr/bin/env bash
# Run the 2021 crop/OCR/LLM pipeline and evaluate the final JSON.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${DOCLING_PYTHON:-$REPO_ROOT/.venv/bin/python}"
ARGS=()
RAW="$REPO_ROOT/dataset/jsons/2021/raw/2021_raw.json"
if [[ -f "$RAW" ]]; then
    ARGS+=(--raw-json "$RAW")
fi
exec "$PYTHON" -u -m experiments.vision \
    "$REPO_ROOT/dataset/pdfs/2021.pdf" "${JSON_CROPS_OUTPUT:-$REPO_ROOT/outputs/json_crops}" \
    --golden-json "$REPO_ROOT/dataset/golden/neet_2021/2021_mcqs.json" \
    "${ARGS[@]}" "$@"
