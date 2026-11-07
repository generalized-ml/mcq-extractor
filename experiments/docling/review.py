"""Original Docling evidence and vision review experiment."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from mcq_extractor.review_api import (PROMPT, SCHEMA, build_evidence, call_openrouter,
    resolve, validate_response, without_images)
from mcq_extractor.common import save_json


def run(args: argparse.Namespace) -> None:
    raw_bytes = args.raw_json.read_bytes()
    mcq_bytes = args.mcq_json.read_bytes()
    raw = json.loads(raw_bytes)
    document = raw.get("document", raw)
    original = json.loads(mcq_bytes)
    tasks = [(f"{bucket}/{index}", q) for bucket in ("questions", "review_candidates")
             for index, q in enumerate(original.get(bucket, []))
             if args.review_all or bucket == "review_candidates" or q.get("needs_review") or q.get("need_review")]
    identity = {
        "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "mcqs_sha256": hashlib.sha256(mcq_bytes).hexdigest(),
        "model": args.model, "max_images": args.max_images, "prompt_version": 2,
        "review_all": args.review_all,
    }
    print(f"Selected {len(tasks)} questions for LLM extraction.", flush=True)
    if args.dry_run:
        plans = []
        for key, question in tasks[:args.limit] if args.limit is not None else tasks:
            evidence, images = build_evidence(document, question, args.max_images)
            plans.append({"key": key, "question_number": question["question_number"],
                          "images": evidence["attached_image_refs"],
                          "source_count": len(evidence["sources"]),
                          "warnings": evidence["evidence_warnings"]})
        save_json(args.output.with_name(args.output.stem + "_plan.json"), {"tasks": plans})
        print(f"Dry run: prepared {len(plans)} evidence bundles; no API calls made.")
        return
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key or not args.model:
        raise ValueError("Set OPENROUTER_API_KEY and OPENROUTER_MODEL (or pass --model).")
    results: dict = {}
    if args.resume and args.output.exists():
        checkpoint = json.loads(args.output.read_text())
        if checkpoint.get("hybrid", {}).get("identity") != identity:
            raise ValueError("Resume inputs/model/settings differ from the saved run.")
        results = checkpoint["hybrid"]["results"]

    def checkpoint() -> None:
        output = copy.deepcopy(original)
        output["questions"] = []
        output["review_candidates"] = []
        for bucket in ("questions", "review_candidates"):
            for index, question in enumerate(original.get(bucket, [])):
                record = results.get(f"{bucket}/{index}", {})
                entry = copy.deepcopy(question)
                target = bucket
                if record.get("status") == "success":
                    extracted = record["extraction"]
                    entry.update({field: extracted[field] for field in
                                  ("question_text", "options", "correct_answer", "image_refs")})
                    entry["warnings"] = extracted["warnings"] + record["evidence_warnings"]
                    entry["needs_review"] = extracted["status"] != "complete" or bool(entry["warnings"])
                    entry.pop("need_review", None)
                    entry["extraction_method"] = "openrouter_vision"
                    entry["heuristic_warnings"] = question.get("warnings", [])
                    entry["llm"] = {k: record[k] for k in ("model", "response_id", "usage")}
                    # Incomplete model responses remain in their original bucket.
                    if extracted["status"] == "complete" and len(entry["options"]) >= 2:
                        target = "questions"
                elif record.get("status") == "error":
                    entry["needs_review"] = True
                    entry["llm_error"] = record["error"]
                output[target].append(entry)
        output["questions"].sort(key=lambda q: (min(q.get("pages") or [0]),
            int(q["question_number"]) if str(q["question_number"]).isdigit() else 0))
        output["question_count"] = len(output["questions"])
        output["extraction_method"] = "heuristics_with_openrouter_review"
        output["hybrid"] = {"identity": identity, "results": results,
                            "source_raw_json": str(args.raw_json.resolve()),
                            "selected_count": len(tasks)}
        save_json(args.output, output)
        if args.output.parent.name == 'mcq_with_llm':
            save_json(args.output.parent.parent / 'final_mcqs.json', output)

    checkpoint()
    calls = 0
    for task_key, question in tasks:
        if results.get(task_key, {}).get("status") == "success":
            continue
        if args.limit is not None and calls >= args.limit:
            break
        calls += 1
        print(f"Extracting question {question['question_number']} ({calls})", flush=True)
        try:
            evidence, images = build_evidence(document, question, args.max_images)
            response = call_openrouter(evidence, images, question, api_key=key,
                model=args.model, retries=args.retries, timeout=args.timeout)
            results[task_key] = {"status": "success", **response,
                                 "evidence_warnings": evidence["evidence_warnings"]}
        except (RuntimeError, ValueError, KeyError, TypeError, IndexError) as exc:
            results[task_key] = {"status": "error", "error": str(exc)}
            print(f"Question {question['question_number']} retained for review: {exc}", flush=True)
        checkpoint()
    print(f"Saved hybrid output: {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_json", type=Path)
    parser.add_argument("mcq_json", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", default=os.environ.get("OPENROUTER_MODEL"))
    parser.add_argument("--limit", type=int, help="Maximum new question requests this run")
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--review-all", action="store_true",
                        help="Send every question for vision review, including unflagged formulas")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.output = args.output or args.mcq_json.with_name(args.mcq_json.stem + "_hybrid.json")
    if args.output.resolve() in {args.raw_json.resolve(), args.mcq_json.resolve()}:
        parser.error("Output must differ from both inputs.")
    if args.max_images < 1 or args.retries < 0 or args.timeout <= 0 or (args.limit is not None and args.limit < 0):
        parser.error("Invalid image limit, retries, timeout or question limit.")
    try:
        run(args)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        parser.exit(1, f"Hybrid extraction failed: {exc}\n")


if __name__ == "__main__":
    main()
