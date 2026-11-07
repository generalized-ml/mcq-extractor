"""Reuse the existing Docling crop pipeline and snapshot its verified images."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from collections import defaultdict

from mcq_extractor.document.crop_pipeline import build_parser, run as run_crops
from mcq_extractor.common import file_hash


def inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Crop path escapes its directory: {relative}")
    return path


def prepare_crops(pdf: Path, root: Path, args) -> tuple[list[dict], dict]:
    if args.crops_dir:
        source = args.crops_dir.expanduser().resolve()
    else:
        stage = root / "docling_crops"
        existing = stage / pdf.stem / "crop_pipeline.json"
        source = None
        if existing.is_file():
            prior = json.loads(existing.read_text())
            if (prior.get("identity", {}).get("pdf_sha256") == file_hash(pdf)
                    and len(prior.get("results", {})) == len(prior.get("anchors", []))):
                source = existing.parent
        if source is None:
            values = [str(pdf), str(stage)]
            values.extend(['--ocr-backend', args.crop_ocr, '--ocr-device', args.crop_device])
            if args.local_files_only:
                values.append('--local-files-only')
            if args.raw_json:
                values.extend(["--raw-json", str(args.raw_json.expanduser().resolve())])
            if existing.is_file():
                values.append("--resume")
            source = run_crops(build_parser().parse_args(values))
    manifest = json.loads((source / "crop_pipeline.json").read_text())
    if manifest.get("identity", {}).get("pdf_sha256") != file_hash(pdf):
        raise ValueError("Crop manifest belongs to a different PDF")
    records = manifest.get("results", {})
    if not records:
        raise ValueError("Crop manifest has no completed questions")
    candidates = defaultdict(list)
    for key, record in sorted(records.items()):
        candidates[str(record["question"]["question_number"])].append((key, record))
    selected = {}
    discarded = []
    for number, entries in candidates.items():
        if len(entries) == 1:
            selected[entries[0][0]] = entries[0][1]
            continue
        # Cover-page numbered instructions can restart at question 1. A
        # printed answer identifies the real question; if OCR missed it, four
        # parsed options can still distinguish it from an instruction item.
        # Ambiguous cases still fail rather than choose arbitrarily.
        answered = [(key, record) for key, record in entries
                    if record["question"].get("correct_answer") not in (None, "")]
        complete = [(key, record) for key, record in entries
                    if len(record["question"].get("options", [])) >= 4]
        selected_entry = answered[0] if len(answered) == 1 else (
            complete[0] if not answered and len(complete) == 1 else None)
        if selected_entry is None:
            raise ValueError(f"Ambiguous duplicate question number in crop manifest: {number}")
        selected[selected_entry[0]] = selected_entry[1]
        discarded.extend(key for key, _ in entries if key != selected_entry[0])
    plan = []
    for key, record in sorted(selected.items()):
        question = record["question"]
        number = str(question["question_number"])
        patch = inside(source, question["image_refs"][0])
        parts = sorted(path for path in record["artifacts"]
                       if Path(path).parent == patch.relative_to(source).parent
                       and Path(path).name.startswith("part_") and Path(path).suffix == ".png")
        if not parts:
            raise ValueError(f"No source image parts for question {number}")
        folder = root / "questions" / key
        folder.mkdir(parents=True, exist_ok=True)
        artifacts, copied_parts = {}, []
        for relative in [str(patch.relative_to(source)), *parts]:
            original = inside(source, relative)
            digest = file_hash(original)
            if digest != record["artifacts"].get(relative):
                raise ValueError(f"Crop image changed since checkpoint: {original}")
            target = folder / original.name
            shutil.copy2(original, target)
            target_relative = str(target.relative_to(root))
            artifacts[target_relative] = digest
            if relative in parts:
                copied_parts.append(target_relative)
        # Old OCR parsing warnings are not evidence of a bad crop. Retain only
        # actual boundary/continuation failures in the new routing decision.
        crop_warnings = [warning for warning in question.get("warnings", [])
                         if "--max-regions" in warning or "--max-steps" in warning]
        plan.append({"key": key, "question_number": number, "parts": copied_parts,
                     "patch": str((folder / "patch.png").relative_to(root)),
                     "crop_warnings": crop_warnings, "artifacts": artifacts,
                     "crops": question.get("crops", []), "pages": question.get("pages", []),
                     "source_refs": question.get("source_refs", [])})
    numeric = {int(number) for number in candidates if number.isdigit()}
    expected = set(range(1, max(numeric) + 1)) if numeric and len(numeric) == len(candidates) else set()
    missing = sorted(expected - numeric)
    return plan, {"directory": str(source), "available_questions": len(plan),
                  "anchor_count": len(plan) + len(missing),
                  "missing_question_numbers": missing,
                  "discarded_false_anchors": discarded,
                  "manifest_sha256": file_hash(source / "crop_pipeline.json")}
