"""Docling → shrinking Apple Vision crops → crop-only LLM extraction → evaluation."""

from __future__ import annotations

import base64
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .crop_pipeline import build_parser, run as run_crops, validate_args
from .hybrid_mcq.run import PROMPT, SCHEMA, save_json, validate_response
from .pdf_to_mcqs import file_hash


CROP_PROMPT = PROMPT + """
The supplied images are consecutive crops for ONE question, in reading order.
OCR is an unreliable draft. Read the crop images as the source of truth.
Extract all four options. Normalize labels A/B/C/D to 1/2/3/4 respectively,
including the printed answer. Keep source formulas as LaTeX, without solving.
Also transcribe the printed solution/explanation verbatim into solution;
use an empty string if none is printed. Never generate a solution yourself.
If a crop cuts off an option or explanation, set status=needs_review and say so.
Use attached image references for diagrams and image-only options.
"""
CROP_SCHEMA = copy.deepcopy(SCHEMA)
CROP_SCHEMA["properties"]["solution"] = {"type": "string"}
CROP_SCHEMA["required"].append("solution")


def validate_crop_response(value: dict, question: dict, refs: set[str]) -> dict:
    if not isinstance(value, dict) or not isinstance(value.get("solution"), str):
        raise ValueError("Crop response must include a string solution")
    rest = {key: item for key, item in value.items() if key != "solution"}
    validate_response(rest, question, refs)
    if value["status"] == "complete" and [item["label"] for item in value["options"]] != ["1", "2", "3", "4"]:
        raise ValueError("A complete MCQ must have exactly four normalized options")
    return value


def extract_crop(question: dict, images: list[tuple[str, str]], *, model: str,
                 api_key: str, timeout: float, retries: int, record_attempt,
                 prompt: str = CROP_PROMPT, source_text: str | None = None) -> dict:
    content = [{"type": "text", "text": json.dumps({
        "question_number": question["question_number"],
        "ocr_draft": {key: question.get(key) for key in
                      ("question_text", "options", "correct_answer", "solution", "warnings")},
        "attached_image_refs": [ref for ref, _ in images],
        **({"source_ocr_text": source_text} if source_text is not None else {}),
    }, ensure_ascii=False)}]
    for ref, uri in images:
        content.extend([{"type": "text", "text": f"Source crop: {ref}"},
                        {"type": "image_url", "image_url": {"url": uri}}])
    body = {"model": model, "messages": [{"role": "system", "content": prompt},
                                          {"role": "user", "content": content}],
            "max_tokens": 8000, "usage": {"include": True},
            "provider": {"require_parameters": True},
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "cropped_mcq", "strict": True, "schema": CROP_SCHEMA}}}
    for attempt in range(retries + 1):
        request = Request("https://openrouter.ai/api/v1/chat/completions",
                          data=json.dumps(body).encode(), method="POST",
                          headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
        record = {"question_number": question["question_number"], "attempt": attempt + 1,
                  "model": model, "status": "error", "usage": {}}
        retryable = True
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
            record.update(usage=payload.get("usage") or {}, response_id=payload.get("id"),
                          model=payload.get("model", model))
            if "error" in payload:
                raise ValueError("Provider returned an error response")
            choice = payload["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise ValueError("Model response was truncated or refused")
            extraction = validate_crop_response(json.loads(choice["message"]["content"]), question,
                                               {ref for ref, _ in images})
            record["status"] = "success"
            record_attempt(record)
            return {"extraction": extraction, "usage": record["usage"],
                    "response_id": record.get("response_id"), "model": record["model"]}
        except HTTPError as exc:
            exc.close()
            error = f"OpenRouter HTTP {exc.code}"
            retryable = exc.code in {408, 429, 500, 502, 503, 504}
        except (URLError, TimeoutError):
            error = "OpenRouter network request failed or timed out"
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            error = f"Invalid crop response: {exc}"
        record["error"] = error
        record_attempt(record)
        if not retryable or attempt == retries:
            raise RuntimeError(error)
        time.sleep(min(2 ** attempt, 16))
    raise RuntimeError("Crop extraction retries exhausted")


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
    from dotenv import load_dotenv
    load_dotenv(Path.cwd() / ".env", override=False)
    parser = build_parser()
    parser.description = __doc__
    parser.add_argument("--model", default=os.environ.get("OPENROUTER_MODEL") or "google/gemini-2.5-flash-lite")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--golden-json", type=Path, help="Evaluate final JSON against this golden dataset")
    parser.add_argument("--eval-output", type=Path, default=Path(__file__).resolve().parents[2] / "results")
    parser.add_argument("--detect-only", action="store_true", help="Only build OCR crops; no LLM calls or evaluation")
    args = parser.parse_args()
    validate_args(args, parser)
    if args.timeout <= 0 or args.retries < 0:
        parser.error("Timeout must be positive and retries nonnegative")
    if not args.detect_only and not os.environ.get("OPENROUTER_API_KEY"):
        parser.error("Set OPENROUTER_API_KEY in your environment or ignored .env file, or use --detect-only")
    try:
        root = run_crops(args)
        if args.detect_only:
            return
        final = run_llm(root, args)
        print(f"Final JSON: {final}")
        if args.golden_json:
            subprocess.run([sys.executable, "-m", "evaluation", "--mcq-json", str(final),
                            "--experiment-name", "pdf_to_json", "--golden-json", str(args.golden_json), "--output-dir", str(args.eval_output),
                            "--device", "cpu", "--local-files-only"], check=True)
    except (ImportError, OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"json_crops failed: {exc}\n")


if __name__ == "__main__":
    main()
