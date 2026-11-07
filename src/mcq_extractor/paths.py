"""Workspace paths for source checkouts and installed distributions."""

import os
from pathlib import Path


def workspace_root() -> Path:
    """Use an explicit workspace, the source checkout, or the current directory."""
    override = os.environ.get("MCQ_WORKSPACE")
    if override:
        return Path(override).expanduser().resolve()
    source_root = Path(__file__).resolve().parents[2]
    if (source_root / "pyproject.toml").is_file() and (source_root / "src/mcq_extractor").is_dir():
        return source_root
    return Path.cwd().resolve()


REPOSITORY_ROOT = workspace_root()
