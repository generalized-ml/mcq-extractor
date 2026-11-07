"""Shared OpenRouter crop and text review requests."""

from __future__ import annotations

import copy
import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .review_api import PROMPT, SCHEMA, validate_response


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


