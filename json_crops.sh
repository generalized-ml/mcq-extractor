#!/usr/bin/env bash
# Run the 2021 crop/OCR/LLM pipeline and evaluate the final JSON.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${DOCLING_PYTHON:-$SCRIPT_DIR/.venv/bin/python}"
ARGS=()
RAW="$SCRIPT_DIR/dataset/jsons/2021/raw/2021_raw.json"
if [[ -f "$RAW" ]]; then
    ARGS+=(--raw-json "$RAW")
fi
exec "$PYTHON" -u -m pdf_to_jsons.json_crops \
    "$SCRIPT_DIR/dataset/pdfs/2021.pdf" "${JSON_CROPS_OUTPUT:-$SCRIPT_DIR/outputs/json_crops}" \
    --golden-json "$SCRIPT_DIR/dataset/golden/neet_2021/2021_mcqs.json" \
    "${ARGS[@]}" "$@"
