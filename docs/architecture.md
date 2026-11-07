# Original Docling experiment architecture

The `experiments/docling` package contains the original PDF-to-question pipeline.
The maintained native OCR pipeline lives in `src/mcq_extractor`; both versions
reuse `mcq_extractor.document` for conversion and crop generation.
Application code uses package-relative imports; `python -m experiments.docling` and
the installed `experiment-docling` command invoke the same orchestrator.

1. `run_converter.py` loads the local `docling/` checkout when available, enables
   OCR, table structure, page images, picture images, and parsed pages, and saves
   the public Docling `ConversionResult` as JSON.
2. `extract_mcqs.py` walks document elements and emits question text, options,
   source references, page numbers, and incomplete review candidates.
3. `experiments/docling/review.py` builds evidence from source elements and embedded images,
   calls OpenRouter for selected questions, validates responses, and checkpoints
   results. Dry runs save an evidence plan without API calls.

`experiments/docling/pipeline.py` coordinates these stages. A per-PDF manifest records source and
code hashes; stage output hashes protect resumed conversion and extraction.
Hybrid checkpoints track review inputs and settings independently. Failures keep
partial results available for inspection and retry.

`mcq_extractor.evaluation` compares predictions with golden questions by ID. It reports exact
word-prefix scores and local sentence embedding similarities. `show_mcq.py`
renders a selected question and source evidence for manual inspection.

## Data and ownership

- `dataset/` holds the three requested directories copied from NEET's `datasets/`.
  File contents remain unchanged, including historical absolute provenance paths.
- `outputs/` is the recommended destination for fresh conversion runs.
- `results/` holds evaluation reports.
- `tests/fixtures/` preserves the former `testing/data/` files.
- `docling/` retains its own Git repository and pre-existing local changes.

The application source and datasets came from NEET's conversion and evaluation
modules. Unrelated exam generation, web scraping, MinerU, virtual environments,
logs, and caches were excluded. Existing sandbox tools were consolidated into the
package and its offline tests; the old sandbox was backed up outside the workspace.
