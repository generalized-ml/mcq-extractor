# Publishing MCQ Extractor

The PyPI distribution is `mcq-extractor`; the import package is `mcq_extractor`.
Only the maintained package is distributed. Experimental pipelines, datasets,
model weights, checkpoints, and credentials stay outside the wheel and sdist.

## Validate a release

```bash
python -m pip install -e '.[dev]'
python -m unittest discover -s tests -p 'test_*.py'
python -m build
python -m twine check --strict dist/*
```

Version `0.1.0` is declared in `pyproject.toml` and
`src/mcq_extractor/__init__.py`. Update both for subsequent releases. The
GitHub release tag must match, for example `v0.1.0`.

## One-time PyPI configuration

Log into PyPI and create a pending publisher at
https://pypi.org/manage/account/publishing/ with these exact values:

| Field | Value |
| --- | --- |
| PyPI project | `mcq-extractor` |
| GitHub owner | `generalized-ml` |
| GitHub repository | `mcq-extractor` |
| Workflow filename | `publish.yml` |
| Environment | `pypi` |

No PyPI API token needs to be stored in GitHub. Name availability is not reserved
until the first successful publication.

## Publish

1. Merge the packaging changes into `main` and wait for the Python package CI
   checks to pass.
2. Configure the pending PyPI publisher above before publishing the draft release.
3. Publish the GitHub release tagged `v0.1.0`.
4. Watch the **Publish Python package** workflow. It validates and builds the
   distributions, publishes them through PyPI Trusted Publishing, and attaches
   them to the GitHub release.
5. Verify installation in a fresh environment with
   `python -m pip install mcq-extractor==0.1.0` and `mcq-extract --help`.

If the first publish fails only because the pending publisher was missing, add
it and rerun the failed workflow job. PyPI versions cannot be overwritten; use a
new version for changes after a successful upload.

Reference: [PyPI Trusted Publishing documentation](https://docs.pypi.org/trusted-publishers/using-a-publisher/).
