# Shared evaluation

Run `PYTHONPATH=src .venv/bin/python -m evaluation --help`.

See [evaluation documentation](../../docs/evaluation.md) for commands and scoring.
Results append to `results/experiment.csv`, with a pipeline label and unique run number.
Each method's recorded PDF wall time and evaluation duration are saved in separate
columns. Pipeline runs save timing sessions in per-PDF `runtime.json` sidecars;
missing historical timings stay blank, and combined runs report timing coverage.
Existing compatible CSVs gain timing columns without recomputing old scores.

The plot command is `PYTHONPATH=src .venv/bin/python -m evaluation.plot_experiments`.
It reads the shared CSV and saves `results/experiments.png`.
