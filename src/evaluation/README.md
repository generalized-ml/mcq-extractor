# Shared evaluation

Run `PYTHONPATH=src .venv/bin/python -m evaluation --help`.

See [evaluation documentation](../../docs/evaluation.md) for commands and scoring.
Results append to `results/experiment.csv`, with a pipeline label and unique run number.

The plot command is `PYTHONPATH=src .venv/bin/python -m evaluation.plot_experiments`.
It reads the shared CSV and saves `results/experiments.png`.
