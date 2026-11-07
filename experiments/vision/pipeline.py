"""Crop-only vision LLM extraction experiment."""

import base64
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from mcq_extractor.document.crop_pipeline import build_parser, run as run_crops, validate_args
from mcq_extractor.crop_review import CROP_PROMPT, CROP_SCHEMA, extract_crop
from mcq_extractor.common import file_hash, save_json
from mcq_extractor.paths import REPOSITORY_ROOT


def run_llm(root: Path, args) -> Path:
    source = root / "mcqs.json"
    draft = json.loads(source.read_text())
    path = root / "final_mcqs.json"
    # Source grows during bounded/resumed OCR runs; identity binds each crop below.
    identity = {"model": args.model, "prompt": CROP_PROMPT, "schema": CROP_SCHEMA,
                "source_pdf_sha256": file_hash(args.pdf.expanduser().resolve())}
    output = {**draft, "questions": [], "extraction_method": "docling_adaptive_crops_vision_llm",
              "llm_calls": [], "hybrid": {"identity": identity, "results": {}}}
    if path.exists():
        if not args.resume:
            raise ValueError("Final crop JSON exists; use --resume")
        output = json.loads(path.read_text())
        if output["hybrid"]["identity"] != identity:
            raise ValueError("LLM model, prompt, or PDF differs from saved run")
    results = output["hybrid"]["results"]

    def checkpoint():
        questions = []
        for index, question in enumerate(draft["questions"]):
            entry = copy.deepcopy(question)
            record = results.get(str(index), {})
            if record.get("status") == "success":
                entry.update(record["extraction"])
                entry["ocr_warnings"] = question["warnings"]
                entry["needs_review"] = entry["status"] != "complete" or bool(entry["warnings"])
                entry["llm"] = {key: record[key] for key in ("usage", "response_id", "model")}
            else:
                entry["needs_review"] = True
                entry["warnings"] = [*question["warnings"], record.get("error", "LLM extraction pending.")]
            questions.append(entry)
        output.update(questions=questions, question_count=len(questions),
                      anchor_count=draft["anchor_count"], pending_count=draft["pending_count"],
                      llm_pending_count=sum(str(i) not in results or results[str(i)]["status"] != "success"
                                            for i in range(len(questions))))
        save_json(path, output)

    checkpoint()
    processed = 0
    for index, question in enumerate(draft["questions"]):
        # Patch files are the evidence source; images contain no API credentials.
        parts = sorted((root / Path(question["image_refs"][0]).parent).glob("part_*.png"))
        if not parts:
            raise ValueError(f"No crop images for question {question['question_number']}")
        crop_hashes = {str(part.relative_to(root)): file_hash(part) for part in parts}
        previous = results.get(str(index))
        if previous and previous.get("status") == "success":
            if previous["crop_sha256"] != crop_hashes:
                raise ValueError("Saved LLM source crops changed")
            continue
        if args.limit is not None and processed >= args.limit:
            break
        print(f"LLM extraction: question {question['question_number']}", flush=True)
        images = [(f"#/crops/{i}", "data:image/png;base64," + base64.b64encode(part.read_bytes()).decode())
                  for i, part in enumerate(parts)]

        def record_attempt(record):
            output["llm_calls"].append(record)
            checkpoint()

        try:
            response = extract_crop(question, images, model=args.model,
                                    api_key=os.environ["OPENROUTER_API_KEY"], timeout=args.timeout,
                                    retries=args.retries, record_attempt=record_attempt)
            results[str(index)] = {"status": "success", **response, "crop_sha256": crop_hashes}
        except RuntimeError as exc:
            results[str(index)] = {"status": "error", "error": str(exc), "crop_sha256": crop_hashes}
        checkpoint()
        processed += 1
    if any(record["status"] == "error" for record in results.values()):
        raise RuntimeError(f"Some crop LLM calls failed; results saved in {path}. Use --resume to retry.")
    return path


def main() -> None:
    from mcq_extractor.runtime import RuntimeRecorder
    started = time.perf_counter()
    from dotenv import load_dotenv
    load_dotenv(Path.cwd() / ".env", override=False)
    parser = build_parser()
    parser.description = __doc__
    parser.add_argument("--model", default=os.environ.get("OPENROUTER_MODEL") or "google/gemini-2.5-flash-lite")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--golden-json", type=Path, help="Evaluate final JSON against this golden dataset")
    parser.add_argument("--eval-output", type=Path, default=REPOSITORY_ROOT / "results")
    parser.add_argument("--detect-only", action="store_true", help="Only build OCR crops; no LLM calls or evaluation")
    args = parser.parse_args()
    validate_args(args, parser)
    if args.timeout <= 0 or args.retries < 0:
        parser.error("Timeout must be positive and retries nonnegative")
    if not args.detect_only and not os.environ.get("OPENROUTER_API_KEY"):
        parser.error("Set OPENROUTER_API_KEY in your environment or ignored .env file, or use --detect-only")
    try:
        root = run_crops(args)
        runtime = RuntimeRecorder(root, 'pdf_to_json', started=started, history_complete=not args.resume)
        runtime.save()
        if args.detect_only:
            runtime.save(finished=True)
            return
        final = run_llm(root, args)
        runtime.save(finished=True)
        print(f"Final JSON: {final}")
        if args.golden_json:
            subprocess.run([sys.executable, "-m", "mcq_extractor.evaluation", "--mcq-json", str(final),
                            "--experiment-name", "pdf_to_json", "--golden-json", str(args.golden_json), "--output-dir", str(args.eval_output),
                            "--device", "cpu", "--local-files-only"], check=True)
    except (ImportError, OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"json_crops failed: {exc}\n")


if __name__ == "__main__":
    main()
