#!/usr/bin/env bash
# Convert dataset/pdfs to dataset/jsons by default.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONVERTER="$SCRIPT_DIR/src/pdf_to_jsons/pdf_to_mcqs.py"
export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Usage: $0 [PDF_DIRECTORY] [OUTPUT_DIRECTORY] [PIPELINE_OPTIONS...]"
    echo "Default input: $SCRIPT_DIR/dataset/pdfs"
    echo "Default output: $SCRIPT_DIR/dataset/jsons"
    echo "Processes all PDFs in the input directory."
    echo "Writes per-PDF raw/, mcqs/, mcq_with_llm/, and pipeline.json under OUTPUT_DIRECTORY."
    echo "Pass pipeline options after both directories, e.g. --dry-run or --resume."
    echo "The LLM stage requires OPENROUTER_API_KEY unless --dry-run is passed."
    echo "Set DOCLING_PYTHON to select a Python executable."
    exit 0
fi

PDF_DIR="${1:-$SCRIPT_DIR/dataset/pdfs}"
if [[ ! -d "$PDF_DIR" ]]; then
    echo "PDF directory not found: $PDF_DIR" >&2
    exit 1
fi
if [[ ! -f "$CONVERTER" ]]; then
    echo "Converter not found: $CONVERTER" >&2
    exit 1
fi
PDF_DIR="$(cd -- "$PDF_DIR" && pwd)"
OUTPUT_DIR="${2-$SCRIPT_DIR/dataset/jsons}"
if [[ -z "$OUTPUT_DIR" ]]; then
    echo "Output directory cannot be empty." >&2
    exit 2
fi
mkdir -p -- "$OUTPUT_DIR"
OUTPUT_DIR="$(cd -- "$OUTPUT_DIR" && pwd)"
if (( $# >= 2 )); then
    shift 2
elif (( $# == 1 )); then
    shift
fi
PIPELINE_ARGS=("$@")

if [[ -n "${DOCLING_PYTHON:-}" ]]; then
    PYTHON="$DOCLING_PYTHON"
elif [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
    PYTHON="$SCRIPT_DIR/.venv/bin/python"
else
    PYTHON="python3"
fi
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "Python executable not found: $PYTHON" >&2
    exit 1
fi
if ! "$PYTHON" -m pdf_to_jsons --help >/dev/null; then
    echo "Converter setup failed; resolve the error above before retrying." >&2
    exit 1
fi
DRY_RUN=0
for ARG in ${PIPELINE_ARGS[@]+"${PIPELINE_ARGS[@]}"}; do
    if [[ "$ARG" == "--dry-run" ]]; then
        DRY_RUN=1
    fi
done
if (( DRY_RUN == 0 )) && [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "OPENROUTER_API_KEY is required for LLM review. Export it before running," >&2
    echo "or pass both directories followed by --dry-run to skip API calls." >&2
    exit 1
fi

shopt -s nullglob
PDFS=("$PDF_DIR"/*.[pP][dD][fF])
TOTAL=0
SUCCEEDED=0
FAILED=()
for PDF in "${PDFS[@]}"; do
    [[ -f "$PDF" ]] || continue
    TOTAL=$((TOTAL + 1))
    echo "Converting: $PDF"
    if "$PYTHON" -u -m pdf_to_jsons "$PDF" "$OUTPUT_DIR" ${PIPELINE_ARGS[@]+"${PIPELINE_ARGS[@]}"}; then
        SUCCEEDED=$((SUCCEEDED + 1))
    else
        FAILED+=("$PDF")
        echo "FAILED: $PDF (continuing with remaining PDFs)" >&2
    fi
done

if (( TOTAL == 0 )); then
    echo "No PDFs to convert in: $PDF_DIR" >&2
    exit 1
fi
echo "Converted $SUCCEEDED/$TOTAL PDFs."
if (( ${#FAILED[@]} > 0 )); then
    printf 'Failed PDF: %s\n' "${FAILED[@]}" >&2
    exit 1
fi
