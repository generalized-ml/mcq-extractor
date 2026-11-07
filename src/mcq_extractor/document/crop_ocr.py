"""Portable local crop OCR and conservative text-to-MCQ parsing."""

from __future__ import annotations

import re
import math
import sys
from typing import Any


OPTION_TOKEN = re.compile(r"(?<!\w)(?:\(([1-4A-Da-d])\)|([A-Da-d])[.)](?=\s)|([1-4])[)](?=\s))")
ANSWER_TOKEN = re.compile(r"\b(?:correct\s+answer|answer|ans\.?)\s*[:.\-]?\s*\(?([1-4A-Da-d])\)?(?:\s|$|[.,])", re.I)
SOLUTION_TOKEN = re.compile(r"(?:^|\n)\s*(?:sol\.|solution\b|explanation\b)\s*[:.\-]?", re.I)


def canonical_label(label: str) -> str:
    return str(ord(label.upper()) - ord("A") + 1) if label.upper() in "ABCD" else label


class MacVisionOCR:
    """Use ocrmac, the Apple Vision wrapper also used by Docling OcrMacModel."""

    def __init__(self, languages: list[str] | None = None):
        if sys.platform != "darwin":
            raise RuntimeError("Apple Vision OCR requires macOS.")
        try:
            from ocrmac import ocrmac
        except ImportError as exc:
            raise RuntimeError("Install crop dependencies: python -m pip install -e '.[crop]'") from exc
        self.reader = ocrmac.OCR
        self.languages = languages or ["en-US"]
        # ocrmac can silently return [] when Vision's request itself failed.
        # Detect that once, rather than saving hundreds of empty question drafts.
        from PIL import Image, ImageDraw, ImageFont
        probe = Image.new("RGB", (500, 80), "white")
        try:
            ImageDraw.Draw(probe).text((12, 12), "Question 123 OCR ready", fill="black", font=ImageFont.load_default(size=30))
            found = self.reader(probe, framework="vision", recognition_level="accurate",
                                language_preference=self.languages).recognize()
        finally:
            probe.close()
        if not found:
            raise RuntimeError("Apple Vision failed its readable-text preflight. Run this command in a normal macOS terminal with Vision access.")

    def __call__(self, image: Any, *, recognition: str = "accurate") -> dict:
        annotations = self.reader(image, framework="vision", recognition_level=recognition,
                                  language_preference=self.languages).recognize(px=True)
        lines = [{"text": str(text), "confidence": float(confidence), "bbox": list(box)}
                 for text, confidence, box in annotations]
        return format_lines(lines, recognition=recognition)


def format_lines(lines: list[dict], *, recognition: str = 'accurate') -> dict:
    # Group fragments on a shared row, then read left to right (inline options).
    rows: list[list[dict]] = []
    for line in sorted(lines, key=lambda item: (item["bbox"][1], item["bbox"][0])):
        center = (line["bbox"][1] + line["bbox"][3]) / 2
        if rows:
            previous = rows[-1][0]["bbox"]
            previous_center = (previous[1] + previous[3]) / 2
            same_row = abs(center - previous_center) < min(previous[3] - previous[1], line["bbox"][3] - line["bbox"][1]) * .5
        else:
            same_row = False
        if same_row:
            rows[-1].append(line)
        else:
            rows.append([line])
    text = "\n".join("  ".join(item["text"] for item in sorted(row, key=lambda item: item["bbox"][0])) for row in rows)
    return {"text": text, "lines": lines, "recognition": recognition,
            "mean_confidence": sum(line["confidence"] for line in lines) / len(lines) if lines else 0.0}


def resolve_ocr_backend(backend: str, *, platform: str | None = None) -> str:
    if backend == 'auto':
        return 'vision' if (platform or sys.platform) == 'darwin' else 'hf'
    if backend not in ('vision', 'hf'):
        raise ValueError(f'Unknown crop OCR backend: {backend}')
    return backend


class HFMobileCropOCR:
    """Adapt native PP-OCRv5 text/line polygons to the crop engine's pixel boxes."""

    def __init__(self, *, device='auto', local_files_only=False):
        try:
            from mcq_extractor.ocr import HFMobileReader
            self.reader = HFMobileReader(device=device, local_files_only=local_files_only)
        except ImportError as exc:
            raise RuntimeError("Install portable crop OCR: python -m pip install -e '.[my,crop]'") from exc
        if self.reader.device == 'cpu':
            import torch
            torch.set_num_threads(2)

    def __call__(self, image: Any, *, recognition: str = 'accurate') -> dict:
        result = self.reader.read_image(image)
        lines = []
        for line in result['lines']:
            points = line['box']
            xs, ys = [float(p[0]) for p in points], [float(p[1]) for p in points]
            if not xs or not all(math.isfinite(v) for v in xs + ys):
                raise ValueError('Native crop OCR returned invalid polygon coordinates')
            score = float(line['score'])
            lines.append({'text': str(line['text']),
                          'confidence': score if math.isfinite(score) else 0.0,
                          'bbox': [min(xs), min(ys), max(xs), max(ys)]})
        return {**format_lines(lines, recognition=recognition), 'backend': 'hf', 'device': self.reader.device}


def create_crop_ocr(backend: str, *, languages=None, device='auto', local_files_only=False):
    backend = resolve_ocr_backend(backend)
    if backend == 'vision':
        return MacVisionOCR(languages)
    return HFMobileCropOCR(device=device, local_files_only=local_files_only)


def parse_question(text: str, number: str, *, require_answer: bool = True) -> dict:
    text = text.strip()
    # The crop anchor identifies the question; never infer another ID from OCR.
    text = re.sub(rf"^\s*(?:Q(?:uestion)?\.?\s*)?{re.escape(number)}[.)]\s*", "", text, count=1, flags=re.I)
    answer = ANSWER_TOKEN.search(text)
    solution = SOLUTION_TOKEN.search(text)
    tail = min((match.start() for match in (answer, solution) if match), default=len(text))
    body = text[:tail]
    markers = list(OPTION_TOKEN.finditer(body))
    options = []
    for index, marker in enumerate(markers):
        label = canonical_label(next(group for group in marker.groups() if group))
        end = markers[index + 1].start() if index + 1 < len(markers) else len(body)
        options.append({"label": label, "text": body[marker.end():end].strip()})
    stem = body[:markers[0].start()].strip() if markers else body.strip()
    warnings = []
    labels = [option["label"] for option in options]
    if not stem:
        warnings.append("Question text is missing.")
    if labels != ["1", "2", "3", "4"]:
        warnings.append("Expected exactly four options in order (1–4 or A–D).")
    if any(not option["text"] for option in options):
        warnings.append("An option has no OCR text; inspect its crop for diagrams or unreadable text.")
    if require_answer and not (answer or solution):
        warnings.append("No printed answer or solution was recognized.")
    solution_text = text[solution.end():].strip() if solution else ""
    if solution and not solution_text:
        warnings.append("Solution heading has no recognized content.")
    return {"question_number": number, "question_text": stem, "options": options,
            "correct_answer": canonical_label(answer.group(1)) if answer else None,
            "solution": solution_text, "warnings": warnings, "needs_review": bool(warnings)}
