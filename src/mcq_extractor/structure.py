"""Deterministic MCQ parsing and explicit low-confidence routing rules."""

from __future__ import annotations

import math
import re

from mcq_extractor.document.crop_ocr import canonical_label

MATH = re.compile(r"\$\$.*?\$\$|(?<!\\)\$[^$]*?(?<!\\)\$|\\\[.*?\\\]|\\\(.*?\\\)|<table\b.*?</table>", re.S | re.I)
MARKER = r"(?:\(([1-9A-Ha-h])\)|([1-9A-Ha-h])[.)])"
LINE_OPTION = re.compile(r"(?m)^[ \t]*(?:[-*][ \t]+)?" + MARKER)
INLINE_OPTION = re.compile(r"(?<!\w)" + MARKER + r"(?=[ \t])")
ANSWER = re.compile(r"(?im)^[ \t]*(?:correct\s+answer|answer|ans\.?)\s*[:.\-]?\s*\(?([1-4A-Da-d])\)?(?:\s|$|[.,])")
SOLUTION = re.compile(r"(?im)^[ \t]*(?:sol\.|solution\b|explanation\b)\s*[:.\-]?")


def masked(text: str) -> str:
    return MATH.sub(lambda match: "".join("\n" if char == "\n" else " " for char in match.group()), text)


def parse_mcq(text: str, number: str, *, require_answer: bool = True) -> dict:
    text = re.sub(r"(?m)^\s*#{1,6}\s+", "", text.strip())
    text = re.sub(r"\*\*([^\n]+?)\*\*", r"\1", text)
    warnings = []
    number_match = re.match(r"^(?:Q(?:uestion)?\.?\s*)?(\d+)[.)]\s*", text, re.I)
    if number_match:
        if number_match.group(1) != str(number):
            warnings.append("OCR question number differs from the Docling anchor.")
        text = text[number_match.end():]
    safe = masked(text)
    answer = ANSWER.search(safe)
    solution = SOLUTION.search(safe)
    end = min((match.start() for match in (answer, solution) if match), default=len(text))
    body, safe_body = text[:end], safe[:end]
    markers = list(LINE_OPTION.finditer(safe_body))

    def label(marker):
        return next(value for value in marker.groups() if value)

    # A matching table may contain A–D in the stem before numeric answer choices.
    numeric = [m for m in markers if label(m).isdigit()]
    if [label(m) for m in numeric] == ["1", "2", "3", "4"]:
        markers = numeric
    if len(markers) != 4:
        inline = list(INLINE_OPTION.finditer(safe_body))
        if [canonical_label(label(m)) for m in inline] == ["1", "2", "3", "4"]:
            markers = inline
        elif inline and not markers:
            warnings.append("Ambiguous inline option boundaries.")
    options = []
    for index, marker in enumerate(markers):
        stop = markers[index + 1].start() if index + 1 < len(markers) else len(body)
        options.append({"label": canonical_label(label(marker)), "text": body[marker.end():stop].strip(),
                        "image_refs": []})
    question_text = body[:markers[0].start()].strip() if markers else body.strip()
    if not question_text:
        warnings.append("Question text is missing.")
    if [option["label"] for option in options] != ["1", "2", "3", "4"]:
        warnings.append("Expected four distinct options in printed order.")
    if any(not option["text"] for option in options):
        warnings.append("An option has no recognized text.")
    if require_answer and answer is None:
        warnings.append("Printed correct answer is missing.")
    solution_text = text[solution.end():].strip() if solution else ""
    if solution and not solution_text:
        warnings.append("Solution heading has no content.")
    if "\ufffd" in text:
        warnings.append("OCR contains replacement characters.")
    return {"question_number": str(number), "question_text": question_text, "options": options,
            "correct_answer": canonical_label(answer.group(1)) if answer else None,
            "solution": solution_text, "image_refs": [], "warnings": warnings,
            "needs_review": bool(warnings)}


def route_question(question: dict, parts: list[dict], *, threshold: float,
                   max_low_token_fraction: float, crop_warnings: list[str] | None = None) -> dict:
    reasons = list(question["warnings"]) + list(crop_warnings or [])
    scores = []
    for index, part in enumerate(parts, 1):
        metrics = part.get("confidence") or {}
        score, low = metrics.get("score"), metrics.get("low_token_fraction")
        if (isinstance(score, bool) or not isinstance(score, (int, float))
                or not math.isfinite(score) or not 0 <= score <= 1):
            reasons.append(f"Crop {index}: OCR confidence unavailable.")
        else:
            scores.append(score)
            if score < threshold:
                reasons.append(f"Crop {index}: OCR score below {threshold:g}.")
        if (not isinstance(low, (int, float)) or isinstance(low, bool)
                or not math.isfinite(low) or not 0 <= low <= 1):
            reasons.append(f"Crop {index}: token uncertainty unavailable.")
        elif low > max_low_token_fraction:
            reasons.append(f"Crop {index}: too many uncertain tokens.")
        if part.get("truncated"):
            reasons.append(f"Crop {index}: OCR output reached the token limit.")
        if not part.get("text", "").strip():
            reasons.append(f"Crop {index}: empty transcription.")
    if not parts:
        reasons.append("No crop transcription available.")
    return {"review_required": bool(reasons), "reasons": list(dict.fromkeys(reasons)),
            "score": min(scores) if len(scores) == len(parts) and scores else None,
            "threshold": threshold, "max_low_token_fraction": max_low_token_fraction,
            "calibrated": False}
