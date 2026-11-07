#!/usr/bin/env python3
"""Extract numbered MCQs from Docling JSON using markers and reading order.

Uses only the Python standard library. Accepts a ConversionResult JSON or a
standalone DoclingDocument JSON. Does not rerun PDF conversion or infer answers.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterator


QUESTION = re.compile(r"^\s*(?:Q(?:uestion)?\.?\s*)?(\d{1,4})[.)]\s+(.*)$", re.I)
QUESTION_MARKER = re.compile(r"^(\d{1,4})[.)]$")
OPTION = re.compile(r"^\s*(?:\(([A-Da-d1-9])\)|([A-Da-d])[.)])\s*(.*)$")
ANSWER = re.compile(
    r"^\s*(?:correct\s+answer|answer|ans\.?)\s*[:.\-]?\s*"
    r"\(?([A-Da-d1-9])\)?(?:\s|$|[.,])", re.I
)
SOLUTION = re.compile(r"^\s*(?:sol\.|solution\b|explanation\b)", re.I)


def iter_items(document: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Follow body references, including nested list groups, in reading order."""
    visited: set[str] = set()

    def walk(item: dict[str, Any]) -> Iterator[dict[str, Any]]:
        ref = item.get("self_ref", "")
        if ref:
            if ref in visited:
                return
            visited.add(ref)
        yield item
        for child in item.get("children", []):
            target: Any = document
            reference = child["cref"]
            if not reference.startswith("#/"):
                raise ValueError(f"Unsupported document reference: {reference}")
            for part in reference[2:].split("/"):
                part = part.replace("~1", "/").replace("~0", "~")
                target = target[int(part)] if isinstance(target, list) else target[part]
            yield from walk(target)

    yield from walk(document["body"])


def extract_mcqs(document: dict[str, Any]) -> dict[str, Any]:
    questions: list[dict[str, Any]] = []
    review_candidates: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    in_solution = False

    def add_source(item: dict[str, Any]) -> None:
        if current is None:
            return
        ref = item.get("self_ref")
        if ref and ref not in current["source_refs"]:
            current["source_refs"].append(ref)
        for provenance in item.get("prov", []):
            page = provenance["page_no"]
            if page not in current["pages"]:
                current["pages"].append(page)

    def finish() -> None:
        if current is None:
            return
        options = current["options"]
        # A numbered paragraph alone could be an instruction or a solution step.
        if len(options) < 2 and current["correct_answer"] is None:
            return
        labels = [option["label"] for option in options]
        warnings = current["warnings"]
        if len(options) != 4:
            warnings.append(f"Found {len(options)} options; check against the source.")
        if len(labels) != len(set(labels)):
            warnings.append("Duplicate option labels.")
        if any(not option["text"] for option in options):
            warnings.append("An option has no extracted text; it may contain an image.")
        if current["correct_answer"] and current["correct_answer"] not in labels:
            warnings.append("Printed answer does not match an extracted option label.")
        if current["picture_refs"] or current["table_refs"]:
            warnings.append("Consult referenced pictures/tables in the input JSON.")
        current["pages"].sort()
        current["needs_review"] = bool(warnings)
        if len(options) >= 2:
            questions.append(current)
        else:
            review_candidates.append(current)

    for item in iter_items(document):
        label = item.get("label", "")
        if label in {"page_header", "page_footer"}:
            continue
        if label in {"picture", "table"}:
            if current is not None and not in_solution:
                add_source(item)
                field = "picture_refs" if label == "picture" else "table_refs"
                current[field].append(item["self_ref"])
            continue
        text = item.get("text", "").strip()
        marker = item.get("marker", "").strip()
        if not text and not marker:
            if current is not None and label == "formula" and not in_solution:
                add_source(item)
                current["warnings"].append("Formula has no extracted text.")
            continue

        # Docling often removes enumeration from text and stores it as marker.
        lines = text.splitlines() or [""]
        for index, line in enumerate(lines):
            prefix = marker if index == 0 else ""
            numbered = QUESTION_MARKER.fullmatch(prefix) if prefix else None
            question_match = QUESTION.match(line) if not prefix else None
            if numbered or question_match:
                finish()
                number = (numbered or question_match).group(1)
                stem = line if numbered else question_match.group(2)
                current = {
                    "question_number": number,
                    "question_text": stem.strip(),
                    "options": [],
                    "correct_answer": None,
                    "pages": [],
                    "source_refs": [],
                    "picture_refs": [],
                    "table_refs": [],
                    "warnings": [],
                }
                in_solution = False
                add_source(item)
                continue
            if current is None:
                continue
            answer = ANSWER.match(line)
            if answer:
                current["correct_answer"] = answer.group(1).upper()
                add_source(item)
                in_solution = True
                continue
            if SOLUTION.match(line):
                in_solution = True
                continue
            if in_solution:
                continue
            option = OPTION.match(f"{prefix} {line}" if prefix else line)
            if option:
                current["options"].append({
                    "label": (option.group(1) or option.group(2)).upper(),
                    "text": option.group(3).strip(),
                })
                add_source(item)
            elif label != "section_header":
                add_source(item)
                if current["options"]:
                    last_option = current["options"][-1]
                    last_option["text"] = f"{last_option['text']} {line}".strip()
                else:
                    current["question_text"] += f"\n{line}"
    finish()
    return {
        "document_name": document.get("name"),
        "extraction_method": "numbered_question_and_option_markers",
        "limitations": [
            "Heuristic extraction; check results against the source.",
            "Inline options on the same line are not split automatically.",
            "Missing OCR text, formulas and image-only options are not reconstructed.",
            "Picture and table references point into the original input document.",
        ],
        "question_count": len(questions),
        "questions": questions,
        "review_candidates": review_candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_json", type=Path, help="Docling conversion JSON")
    parser.add_argument("--output", type=Path, help="Default: <input_stem>_mcqs.json")
    args = parser.parse_args()
    output = args.output or args.input_json.with_name(f"{args.input_json.stem}_mcqs.json")
    if output.resolve() == args.input_json.resolve():
        parser.error("Output must differ from the input JSON.")
    try:
        payload = json.loads(args.input_json.read_text(encoding="utf-8"))
        document = payload.get("document", payload)
        if not isinstance(document, dict) or "body" not in document:
            raise ValueError("Expected a ConversionResult or DoclingDocument JSON.")
        result = extract_mcqs(document)
        result["source_json"] = str(args.input_json.resolve())
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        parser.exit(1, f"Extraction failed: {exc}\n")
    review_count = sum(q["needs_review"] for q in result["questions"])
    print(f"Extracted {result['question_count']} MCQs ({review_count} flagged for review).")
    print(f"Incomplete candidates: {len(result['review_candidates'])}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
