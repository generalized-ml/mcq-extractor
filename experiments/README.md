# Extraction and inference experiments

The maintained pipeline lives in [`src/mcq_extractor`](../src/mcq_extractor).
All commands below run from the repository root after installing the project
with `python -m pip install -e '.[ocr,crop,eval]'`. Install `.[paddle]` or
`.[paddle-mobile]` as needed for the Paddle backends.

| Experiment | Command | Purpose |
| --- | --- | --- |
| Docling | `bash experiments/docling/run.sh --help` | Original conversion, heuristic extraction, and vision review |
| Vision | `bash experiments/vision/run.sh --help` | Crop-only LLM extraction |
| Paddle | `bash experiments/paddle/run.sh --help` | Paddle mobile and vision-language backends |
| Precision | `python -m experiments.precision --help` | Create separate FP16 checkpoints |
| Recognition | `python -m experiments.recognition.parallel_benchmark --help` | Compare serial and multiprocessing recognition |

The three pipeline versions also support `python -m experiments.docling`,
`python -m experiments.vision`, and `python -m experiments.paddle`.
Their launchers establish the repository and source paths automatically.
For direct module execution without an editable install, set `PYTHONPATH=src:.`.

Shared crop generation, review requests, runtime recording, and evaluation live
in the maintained package so experimental versions can reuse the same logic.
Historical output directories and method IDs are preserved for existing reports.
Renamed modules change checkpoint code identities; start a new output directory
when running the reorganized code rather than resuming an older checkpoint.
