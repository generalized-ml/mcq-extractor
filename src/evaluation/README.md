# Shared evaluation

Run `PYTHONPATH=src .venv/bin/python -m evaluation --help`.

See [evaluation documentation](../../docs/evaluation.md) for commands and scoring.
Results append to `results/experiment.csv`, with a pipeline label and unique run number.
Each method's recorded PDF wall time and evaluation duration are saved in separate
columns. Pipeline runs save timing sessions in per-PDF `runtime.json` sidecars;
missing historical timings stay blank, and combined runs report timing coverage.
Existing compatible CSVs gain timing columns without recomputing old scores.
Costs are reported as average USD per PDF (`llm_cost_per_pdf_usd`), with counts of
PDFs with known and unknown costs. The chart uses this per-PDF metric too.

The plot command is `PYTHONPATH=src .venv/bin/python -m evaluation.plot_experiments`.
It reads the shared CSV and saves `results/experiments.png`.
