# Shared evaluation

Run `PYTHONPATH=src:. .venv/bin/python -m mcq_extractor.evaluation --help`.

```bash
.venv/bin/python -m mcq_extractor.evaluation outputs/json_crops_my_all
.venv/bin/python -m mcq_extractor.evaluation outputs/pdf_to_jsons_all dataset/golden

# Print results without creating or changing experiment.csv:
.venv/bin/python -m mcq_extractor.evaluation outputs/json_crops_my_all --no-save
```

Provide any experiment folder, optionally followed by a golden dataset folder
(default `dataset/golden`). Available PDF outputs are discovered automatically;
each command appends one row. Both pipelines publish `<PDF>/final_mcqs.json`
and `<PDF>/mcqs.json`. One PDF output folder also works. Legacy original-method
paths remain readable. The default experiment label is the input folder name;
`--experiment-name` can override it. Explicit `--years` includes missing years
in the comparison; otherwise only discovered PDFs are evaluated.

See [evaluation documentation](../../../docs/evaluation.md) for commands and scoring.
Results append to `results/experiment.csv`, with a pipeline label and unique run number.
Each method's recorded PDF wall time and evaluation duration are saved in separate
columns. Pipeline runs save timing sessions in per-PDF `runtime.json` sidecars;
missing historical timings stay blank, and combined runs report timing coverage.
Existing compatible CSVs gain timing columns without recomputing old scores.
Costs are reported as average USD per PDF (`llm_cost_per_pdf_usd`), with counts of
PDFs with known and unknown costs. The chart uses this per-PDF metric too.

The plot command is `PYTHONPATH=src:. .venv/bin/python -m mcq_extractor.evaluation.plot_experiments`.
It reads the shared CSV and saves `results/experiments.png`.
