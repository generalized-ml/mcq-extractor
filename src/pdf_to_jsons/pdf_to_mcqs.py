#!/usr/bin/env python3
"""Convert a PDF, extract MCQs, and review flagged questions through OpenRouter."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path

from .extract_mcqs import extract_mcqs
from .hybrid_mcq.run import run as run_hybrid
from .hybrid_mcq.run import save_json
from .run_converter import run_converter


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_pipeline(args: argparse.Namespace) -> Path:
    pdf_path = args.pdf_path.expanduser().resolve()
    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected an existing PDF file: {pdf_path}")
    if not args.dry_run and not os.environ.get("OPENROUTER_API_KEY"):
        raise ValueError("Set OPENROUTER_API_KEY before starting the pipeline.")

    root = args.output_folder.expanduser().resolve() / pdf_path.stem
    raw_path = root / "raw" / f"{pdf_path.stem}_raw.json"
    mcqs_path = root / "mcqs" / f"{pdf_path.stem}_mcqs.json"
    hybrid_path = root / "mcq_with_llm" / f"{pdf_path.stem}_mcqs_with_llm.json"
    manifest_path = root / "pipeline.json"
    scripts_dir = Path(__file__).resolve().parent
    identity = {
        "source_pdf": str(pdf_path),
        "pdf_sha256": file_hash(pdf_path),
        "converter_sha256": file_hash(scripts_dir / "run_converter.py"),
        "extractor_sha256": file_hash(scripts_dir / "extract_mcqs.py"),
    }
    manifest = {"identity": identity}
    if args.resume:
        if not manifest_path.is_file():
            raise ValueError("No pipeline checkpoint found. Run without --resume first.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("identity") != identity:
            raise ValueError("PDF or conversion/extraction code changed. Use a new output folder.")
        for name, path in (("raw", raw_path), ("mcqs", mcqs_path)):
            if manifest.get(f"{name}_sha256") and (
                not path.is_file() or file_hash(path) != manifest[f"{name}_sha256"]
            ):
                raise ValueError(f"Saved {name} JSON changed or is missing. Use a new output folder.")
    elif any(path.exists() for path in (manifest_path, raw_path, mcqs_path, hybrid_path)):
        raise ValueError("Pipeline output already exists. Use --resume or a different output folder.")
    save_json(manifest_path, manifest)

    if not manifest.get("raw_sha256"):
        print("Stage 1/3: PDF conversion", flush=True)
        run_converter(pdf_path, raw_path.parent, output_path=raw_path)
        manifest["raw_sha256"] = file_hash(raw_path)
        save_json(manifest_path, manifest)
    else:
        print(f"Stage 1/3: reusing {raw_path}", flush=True)

    if not manifest.get("mcqs_sha256"):
        print("Stage 2/3: heuristic MCQ extraction", flush=True)
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        if raw.get("status") != "success":
            raise ValueError(f"Conversion status is {raw.get('status')}; inspect {raw_path} before continuing.")
        result = extract_mcqs(raw["document"])
        result.update(source_json=str(raw_path), source_pdf=str(pdf_path),
                      conversion_status=raw["status"])
        save_json(mcqs_path, result)
        print(f"Found {result['question_count']} MCQs and {len(result['review_candidates'])} incomplete candidates.")
        manifest["mcqs_sha256"] = file_hash(mcqs_path)
        save_json(manifest_path, manifest)
        del raw, result
    else:
        print(f"Stage 2/3: reusing {mcqs_path}", flush=True)

    print("Stage 3/3: OpenRouter vision extraction", flush=True)
    run_hybrid(argparse.Namespace(
        raw_json=raw_path, mcq_json=mcqs_path, output=hybrid_path,
        model=args.model, limit=args.limit, max_images=args.max_images,
        retries=args.retries, timeout=args.timeout, resume=args.resume,
        dry_run=args.dry_run, review_all=args.review_all,
    ))
    print(f"Output folder: {root}", flush=True)
    if not args.dry_run:
        final = json.loads(hybrid_path.read_text(encoding="utf-8"))
        results = final["hybrid"]["results"]
        failed = sum(item["status"] == "error" for item in results.values())
        pending = final["hybrid"]["selected_count"] - len(results)
        print(f"Final MCQs: {final['question_count']}; incomplete candidates: {len(final['review_candidates'])}; "
              f"failed LLM requests: {failed}; pending: {pending}")
        if failed:
            raise ValueError("Some LLM requests failed. Outputs are saved; use --resume to retry failures.")
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf_path", type=Path)
    parser.add_argument("output_folder", type=Path)
    parser.add_argument("--model", default=os.environ.get("OPENROUTER_MODEL") or "google/gemini-2.5-flash-lite")
    parser.add_argument("--resume", action="store_true", help="Reuse saved conversion and extraction; resume LLM requests")
    parser.add_argument("--review-all", action="store_true", help="Review all questions, including unflagged formulas")
    parser.add_argument("--limit", type=int, help="Maximum new questions sent to the LLM this run")
    parser.add_argument("--dry-run", action="store_true", help="Run conversion/extraction and prepare LLM evidence without API calls")
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if args.max_images < 1 or args.retries < 0 or args.timeout <= 0 or (args.limit is not None and args.limit < 0):
        parser.error("Invalid image limit, retries, timeout or question limit.")
    logging.basicConfig(level=logging.INFO)
    try:
        run_pipeline(args)
    except ImportError as exc:
        parser.exit(1, f"Missing dependency: {exc}\nInstall with: python -m pip install -e './docling[standard]'\n")
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, IndexError) as exc:
        parser.exit(1, f"Pipeline failed: {exc}\n")


if __name__ == "__main__":
    main()
