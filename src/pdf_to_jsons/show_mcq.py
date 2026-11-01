#!/usr/bin/env python3
"""Print one MCQ and its sources; save an HTML report with referenced images.

Example:
    python -m pdf_to_jsons.show_mcq path/to/mcqs.json 40
    python -m pdf_to_jsons.show_mcq path/to/mcqs.json 40 --raw-json path/to/raw.json
"""

from __future__ import annotations

import argparse
import base64
import binascii
import html
import json
import re
from pathlib import Path
from typing import Any


def resolve(document: dict[str, Any], ref: str) -> Any:
    if not ref.startswith("#/"):
        raise ValueError(f"Unsupported reference: {ref}")
    value: Any = document
    for part in ref[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def find_raw_json(mcqs: dict, input_path: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Raw JSON not found: {path}")
        return path
    hints = [mcqs.get("hybrid", {}).get("source_raw_json"), mcqs.get("source_json")]
    for hint in hints:
        if not hint:
            continue
        path = Path(hint).expanduser()
        candidates = [path, input_path.parent / path, input_path.parent / path.name]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
    raise ValueError("Cannot locate the source document. Pass --raw-json path/to/raw.json.")


def item_lines(ref: str, item: dict) -> list[str]:
    lines = [f"Reference: {ref}", f"Type: {item.get('label', 'page')}"]
    if "marker" in item:
        lines.append(f"Marker: {item['marker']}")
    if "text" in item:
        lines.append(f"Text: {item['text'] or '[empty]'}")
    if item.get("orig") and item["orig"] != item.get("text"):
        lines.append(f"Original text: {item['orig']}")
    for prov in item.get("prov", []):
        lines.append(f"Page: {prov.get('page_no')}; bounding box: {prov.get('bbox')}")
    data = item.get("data") or {}
    for cell in data.get("table_cells", []):
        row, col = cell.get("start_row_offset_idx"), cell.get("start_col_offset_idx")
        lines.append(f"Table cell [row={row}, col={col}]: {cell.get('text', '')}")
    return lines


def collect_sources(document: dict, question: dict, include_pages: bool) -> tuple[list, list[str]]:
    refs = list(dict.fromkeys(
        question.get("source_refs", []) + question.get("picture_refs", [])
        + question.get("table_refs", []) + question.get("image_refs", [])
        + [ref for option in question.get("options", []) for ref in option.get("image_refs", [])]
    ))
    sources: list[tuple[str, dict]] = []
    warnings: list[str] = []
    visited: set[str] = set()

    def visit(ref: str) -> None:
        if ref in visited:
            return
        visited.add(ref)
        try:
            item = resolve(document, ref)
            if not isinstance(item, dict):
                raise ValueError("Reference does not resolve to an object.")
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            warnings.append(f"Cannot resolve {ref}: {exc}")
            return
        sources.append((ref, item))
        for field in ("children", "captions", "references", "footnotes"):
            for child in item.get(field, []):
                if isinstance(child, dict) and child.get("cref"):
                    visit(child["cref"])

    for ref in refs:
        visit(ref)
    if include_pages:
        pages = set(question.get("pages", []))
        for _, item in sources:
            pages.update(prov["page_no"] for prov in item.get("prov", []))
        for page_no in sorted(pages):
            visit(f"#/pages/{page_no}")
    return sources, warnings


def save_image(item: dict, directory: Path, index: int) -> tuple[Path | None, str | None]:
    image = item.get("image") or {}
    uri = image.get("uri")
    if not uri:
        return None, None
    extensions = {"image/png": ".png", "image/jpeg": ".jpg",
                  "image/webp": ".webp", "image/gif": ".gif"}
    match = re.fullmatch(r"data:(image/[^;]+);base64,(.*)", uri, re.DOTALL)
    if match is None or match.group(1) not in extensions:
        return None, "Image is not an embedded PNG/JPEG/WebP/GIF; it was not loaded."
    try:
        content = base64.b64decode(match.group(2), validate=True)
    except (binascii.Error, ValueError):
        return None, "Image contains invalid Base64 data."
    path = directory / f"source_{index:03d}{extensions[match.group(1)]}"
    path.write_bytes(content)
    return path, None


def show_question(document: dict, question: dict, bucket: str, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    lines = [f"Question {question['question_number']} ({bucket})", "",
             question.get("question_text", ""), "", "Options:"]
    for option in question.get("options", []):
        lines.append(f"  ({option['label']}) {option.get('text', '')}")
        for ref in option.get("image_refs", []):
            lines.append(f"      Image: {ref}")
    answer = question.get("correct_answer")
    lines += ["", f"Printed answer: {answer if answer is not None else '[not available]'}",
              f"Pages: {question.get('pages', [])}",
              f"Needs review: {question.get('needs_review', question.get('need_review', False))}"]
    lines += [f"Warning: {warning}" for warning in question.get("warnings", [])]
    if question.get("llm_error"):
        lines.append(f"LLM error: {question['llm_error']}")
    print("\n".join(lines))
    sections = [f"<pre>{html.escape(chr(10).join(lines))}</pre>"]
    sources, warnings = collect_sources(document, question, include_pages=True)
    for index, (ref, item) in enumerate(sources, 1):
        text = "\n".join(item_lines(ref, item))
        print(f"\n{text}")
        sections.append(f"<section><pre>{html.escape(text)}</pre>")
        image, warning = save_image(item, directory, index)
        if image is not None:
            print(f"Image: {image.resolve()}")
            sections.append(f'<a href="{image.name}"><img src="{image.name}" '
                            f'alt="{html.escape(ref, quote=True)}"></a>')
        if warning:
            warnings.append(f"{ref}: {warning}")
        sections.append("</section>")
    for warning in warnings:
        print(f"Warning: {warning}")
        sections.append(f"<p>{html.escape(warning)}</p>")
    report = directory / "index.html"
    report.write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>Question {html.escape(str(question["question_number"]))}</title>'
        '<style>body{max-width:1100px;margin:32px auto;padding:0 20px;'
        'font-family:system-ui}pre{white-space:pre-wrap;overflow-wrap:anywhere;'
        'line-height:1.6}section{border-top:1px solid #ccc;margin-top:24px;'
        'padding-top:12px}img{max-width:100%;height:auto}</style><body>'
        + "\n".join(sections) + '</body></html>', encoding="utf-8"
    )
    print(f"\nView text and images: {report.resolve()}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mcq_json", type=Path)
    parser.add_argument("question_number", help="Printed question number")
    parser.add_argument("--raw-json", type=Path, help="Override source JSON location")
    parser.add_argument("--output-dir", type=Path, help="Directory for HTML and images")
    args = parser.parse_args()
    try:
        input_path = args.mcq_json.expanduser().resolve()
        mcqs = json.loads(input_path.read_text(encoding="utf-8"))
        matches = [(bucket, index, q) for bucket in ("questions", "review_candidates")
                   for index, q in enumerate(mcqs.get(bucket, []))
                   if str(q["question_number"]) == args.question_number]
        if not matches:
            raise ValueError(f"Question {args.question_number} was not found in either list.")
        raw_path = find_raw_json(mcqs, input_path, args.raw_json)
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        document = raw.get("document", raw)
        output_root = (args.output_dir.expanduser().resolve() if args.output_dir
                       else input_path.parent / f"{input_path.stem}_views")
        for bucket, index, question in matches:
            safe_number = re.sub(r"[^A-Za-z0-9_-]", "_", str(question["question_number"]))
            suffix = f"_{bucket}_{index}" if len(matches) > 1 else ""
            show_question(document, question, bucket, output_root / f"question_{safe_number}{suffix}")
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        parser.exit(1, f"Cannot display question: {exc}\n")


if __name__ == "__main__":
    main()
