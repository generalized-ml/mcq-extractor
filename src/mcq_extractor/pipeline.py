"""Docling crops → native Hugging Face mobile OCR → structured MCQs → selective LLM review."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from mcq_extractor.common import save_json
from mcq_extractor.common import file_hash
from mcq_extractor.paths import REPOSITORY_ROOT
from mcq_extractor import document, crop_review
from mcq_extractor.runtime import RuntimeRecorder
from .crops import inside, prepare_crops
from .ocr import DET_MODEL, REC_MODEL, DET_REVISION, REC_REVISION
from .confidence import scope_confidence
from .review import review_question
from .structure import parse_mcq, route_question
from .question_workers import execution_settings, transcribe_questions

WORKSPACE = REPOSITORY_ROOT


def code_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.read_bytes())
    for name in ("crop_pipeline.py", "crop_layout.py", "json_crops.py"):
        digest.update((Path(crop_review.__file__) if name == "json_crops.py" else Path(document.__file__).parent / name).read_bytes())
    return digest.hexdigest()


def run(args, *, reader_factory=None, reviewer=review_question) -> Path:
    started = time.perf_counter()
    pdf = args.pdf.expanduser().resolve()
    if not pdf.is_file() or pdf.suffix.lower() != ".pdf":
        raise ValueError(f"Expected an existing PDF: {pdf}")
    root = args.output.expanduser().resolve() / pdf.stem
    path = root / "pipeline.json"
    identity = {"pdf_sha256": file_hash(pdf), "code_sha256": code_hash(),
                "settings": {key: getattr(args, key) for key in (
                    "det_model", "rec_model", "det_revision", "rec_revision", "device", "batch_size", "confidence_threshold",
                    "max_low_token_fraction", "allow_missing_answer", "review_model", "crop_ocr", "crop_device")},
                "crops_dir": str(args.crops_dir.expanduser().resolve()) if args.crops_dir else None,
                "raw_json": str(args.raw_json.expanduser().resolve()) if args.raw_json else None}
    identity["local_models"] = {}
    for model in (args.det_model, args.rec_model):
        folder = Path(model).expanduser()
        if folder.is_dir():
            identity["local_models"][str(folder.resolve())] = {
                str(p.relative_to(folder)): file_hash(p)
                for p in sorted(folder.rglob("*")) if p.is_file() and p.suffix in (".json", ".safetensors")}
    if args.resume:
        if not path.is_file():
            raise ValueError("No Paddle checkpoint found; start without --resume")
        state = json.loads(path.read_text())
        if state["identity"] != identity:
            previous = {key: value for key, value in state["identity"].items() if key != "code_sha256"}
            current = {key: value for key, value in identity.items() if key != "code_sha256"}
            if not state.get("results") and previous == current:
                # A checkpoint with no OCR results can safely adopt the new
                # code, whether its crop plan was already copied or not.
                state["identity"] = identity
                save_json(path, state)
            else:
                raise ValueError("PDF, code, or settings changed; choose a new output directory")
    else:
        if root.exists() and any(root.iterdir()):
            raise ValueError("Output directory is not empty; use --resume or a new output directory")
        root.mkdir(parents=True, exist_ok=True)
        state = {"identity": identity, "results": {}, "llm_calls": []}
        save_json(path, state)
    runtime = RuntimeRecorder(root, 'pdf_to_json_my', started=started, history_complete=not args.resume)
    if "plan" not in state:
        state["plan"], state["crop_source"] = prepare_crops(pdf, root, args)
        save_json(path, state)
        runtime.save()
    plan, results = state["plan"], state["results"]
    for item in plan:
        for relative, expected in item["artifacts"].items():
            if file_hash(inside(root, relative)) != expected:
                raise ValueError(f"Saved crop changed: {relative}")
    for record in results.values():
        for relative, expected in record.get("artifacts", {}).items():
            if file_hash(inside(root, relative)) != expected:
                raise ValueError(f"Saved OCR artifact changed: {relative}")

    def checkpoint():
        runtime.save()
        drafts, final = [], []
        for item in plan:
            record = results.get(item["key"])
            if not record:
                continue
            draft = record["draft"]
            drafts.append(draft)
            question = copy.deepcopy(draft)
            review = record.get("review", {})
            if review.get("status") == "success":
                question.update(review["extraction"])
                question["llm"] = {key: review.get(key) for key in ("usage", "response_id", "model")}
                question["review_method"] = "text_only_llm"
                question["needs_review"] = question["status"] != "complete" or bool(question["warnings"])
                # A text-only reviewer cannot verify pixels absent from its input.
                if record["ocr_uncertain"]:
                    question["warnings"].append("OCR uncertainty remains; text-only review cannot verify the crop image.")
                    question["needs_review"] = True
            elif draft["routing"]["review_required"]:
                question["needs_review"] = True
                question["review_method"] = "pending" if not review else "failed"
                if review.get("error"):
                    question["warnings"].append(review["error"])
            else:
                question["review_method"] = "not_needed"
            final.append(question)
        common = {"source_pdf": str(pdf), "question_count": len(drafts),
                  "anchor_count": state["crop_source"]["anchor_count"],
                  "pending_ocr_count": len(plan) - len(results),
                  "missing_crop_count": max(0, state["crop_source"]["anchor_count"] - len(plan)),
                  "review_candidates": [],
                  "confidence_note": "Uncalibrated OCR confidence plus structural checks; not an accuracy probability."}
        save_json(root / "mcqs.json", {**common, "questions": drafts,
                                       "extraction_method": "hf_mobile_structured", "llm_calls": []})
        save_json(root / "final_mcqs.json", {**common, "questions": final,
            "extraction_method": "hf_mobile_selective_text_review", "llm_calls": state["llm_calls"],
            "llm_selected_count": sum(r["draft"]["routing"]["review_required"] for r in results.values()),
            "llm_pending_count": sum(r["draft"]["routing"]["review_required"]
                                     and r.get("review", {}).get("status") != "success" for r in results.values())})
        save_json(path, state)

    checkpoint()
    pending = [item for item in plan if item['key'] not in results]
    if args.limit is not None:
        pending = pending[:args.limit]
    settings = execution_settings(args, len(pending))
    state['ocr_execution'] = settings
    save_json(path, state)
    if pending:
        print(f"HF mobile OCR: {settings['physical_cores']} physical cores; "
              f"{settings['workers']} workers x {settings['threads_per_worker']} CPU threads; "
              f"{len(pending)} questions (~{settings['questions_per_worker_ceiling']} per worker); "
              f"device={settings['device']}", flush=True)
    options = dict(det_model=args.det_model, rec_model=args.rec_model,
                   det_revision=args.det_revision, rec_revision=args.rec_revision,
                   device=settings['device'], batch_size=args.batch_size,
                   local_files_only=args.local_files_only)
    items_by_key = {item['key']: item for item in pending}
    for completed in transcribe_questions(pending, root, options, settings, reader_factory=reader_factory):
        key = completed['key']
        item = items_by_key[key]
        state['models'] = completed['models']
        print(f"Paddle OCR: question {item['question_number']} ({len(item['parts'])} crop parts)", flush=True)
        folder = root / "questions" / key
        transcripts = completed['transcripts']
        confidence_parts = scope_confidence(transcripts)
        source_text = "\n\n".join(part["text"] for part in transcripts)
        draft = parse_mcq(source_text, item["question_number"], require_answer=not args.allow_missing_answer)
        routing = route_question(draft, confidence_parts, threshold=args.confidence_threshold,
                                 max_low_token_fraction=args.max_low_token_fraction,
                                 crop_warnings=item["crop_warnings"])
        draft.update(routing=routing, needs_review=routing["review_required"],
                     source_crops=item["parts"], patch=item["patch"], crops=item["crops"],
                     pages=item["pages"], source_refs=item["source_refs"])
        save_json(folder / "paddle_ocr.json", {"parts": transcripts, "text": source_text})
        (folder / "paddle_ocr.txt").write_text(source_text, encoding="utf-8")
        save_json(folder / "mcq.json", draft)
        artifacts = {str(p.relative_to(root)): file_hash(p) for p in
                     (folder / "paddle_ocr.json", folder / "paddle_ocr.txt", folder / "mcq.json")}
        # Structural errors may be repaired from text; uncertainty in OCR itself
        # or crop boundaries cannot be eliminated by reviewing text alone.
        ocr_routing = route_question({"warnings": []}, confidence_parts, threshold=args.confidence_threshold,
                                     max_low_token_fraction=args.max_low_token_fraction,
                                     crop_warnings=item["crop_warnings"])
        results[key] = {"draft": draft, "text": source_text, "artifacts": artifacts,
                        "ocr_uncertain": ocr_routing["review_required"],
                        "ocr_execution": completed['execution']}
        checkpoint()
        print(f"  OCR={completed['execution']['seconds']:.3f}s; "
              f"score={routing['score']}; LLM review={routing['review_required']}", flush=True)
    # Worker processes and their OCR models have exited before review/evaluation.
    ordered_results = ((item['key'], results[item['key']]) for item in plan if item['key'] in results)
    selected = [(key, record) for key, record in ordered_results
                if record["draft"]["routing"]["review_required"]
                and record.get("review", {}).get("status") != "success"]
    if selected and not args.ocr_only:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OCR and JSON are saved. Set OPENROUTER_API_KEY and rerun with --resume for selected reviews.")
        for index, (key, record) in enumerate(selected):
            if args.max_llm_questions is not None and index >= args.max_llm_questions:
                break
            print(f"LLM text review: question {record['draft']['question_number']}", flush=True)

            def record_attempt(attempt):
                state["llm_calls"].append({**attempt, "question_key": key})
                checkpoint()

            try:
                response = reviewer(record["draft"], record["text"], model=args.review_model,
                                    api_key=api_key, timeout=args.timeout, retries=args.retries,
                                    record_attempt=record_attempt)
                record["review"] = {"status": "success", **response}
            except RuntimeError as exc:
                record["review"] = {"status": "error", "error": str(exc)}
            checkpoint()
    final = root / "final_mcqs.json"
    runtime.save(finished=True)
    print(f"Saved: {final}")
    if any(record.get("review", {}).get("status") == "error" for record in results.values()) and not args.ocr_only:
        raise RuntimeError("Some LLM reviews failed; saved outputs can be retried with --resume")
    if args.golden_json and not args.ocr_only:
        data = json.loads(final.read_text())
        if data["pending_ocr_count"] or data["llm_pending_count"] or data["missing_crop_count"]:
            print("Evaluation deferred until all crops, OCR, and selected LLM reviews are complete.")
        else:
            subprocess.run([sys.executable, "-m", "mcq_extractor.evaluation", "--mcq-json", str(final),
                            "--experiment-name", "pdf_to_json_my", "--golden-json", str(args.golden_json), "--output-dir", str(args.eval_output),
                            "--device", "cpu", "--local-files-only"], check=True)
    return final


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output", type=Path, default=WORKSPACE / "outputs/native_ocr")
    parser.add_argument("--crops-dir", type=Path, help="Existing verified Docling crop run, e.g. outputs/json_crops_fast/2021")
    parser.add_argument("--raw-json", type=Path, help="Docling source JSON when generating new crops")
    parser.add_argument('--crop-ocr', choices=('auto', 'vision', 'hf'), default='auto',
                        help='Crop generation OCR: Vision on macOS, native PP-OCRv5 on Linux/Colab')
    parser.add_argument('--crop-device', choices=('auto', 'cpu', 'cuda', 'mps'), default='auto')
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, help="Maximum new questions to transcribe this run")
    parser.add_argument("--det-model", default=DET_MODEL, help="HF repo ID or local save_pretrained directory")
    parser.add_argument("--rec-model", default=REC_MODEL, help="HF repo ID or local save_pretrained directory")
    parser.add_argument("--det-revision", default=DET_REVISION)
    parser.add_argument("--rec-revision", default=REC_REVISION)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, help="CPU question workers (default: physical cores // 2); 1 runs serially")
    parser.add_argument("--physical-cores", type=int, help="Override physical core detection when unavailable")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--confidence-threshold", type=float, default=.85)
    parser.add_argument("--max-low-token-fraction", type=float, default=.10)
    parser.add_argument("--allow-missing-answer", action="store_true")
    parser.add_argument("--ocr-only", action="store_true", help="Save local OCR/JSON and routing decisions without LLM calls")
    parser.add_argument("--review-model", default=os.environ.get("OPENROUTER_MODEL") or "google/gemini-2.5-flash-lite")
    parser.add_argument("--max-llm-questions", type=int, help="Limit new question reviews this invocation")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--golden-json", type=Path)
    parser.add_argument("--eval-output", type=Path, default=WORKSPACE / "results")
    return parser


def main():
    from dotenv import load_dotenv
    load_dotenv(WORKSPACE / ".env", override=False)
    parser = build_parser()
    args = parser.parse_args()
    if args.batch_size < 1 or args.retries < 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("Invalid batch size, timeout, or retries")
    if any(value is not None and value < 1 for value in (args.workers, args.physical_cores)):
        parser.error("Workers and physical core count must be positive")
    if any(value is not None and value < 0 for value in (args.limit, args.max_llm_questions)):
        parser.error("Question limits must be nonnegative")
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in (args.confidence_threshold, args.max_low_token_fraction)):
        parser.error("Confidence thresholds must be finite values between 0 and 1")
    try:
        run(args)
    except (ImportError, OSError, RuntimeError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Native OCR pipeline failed: {exc}\n")


if __name__ == "__main__":
    main()
