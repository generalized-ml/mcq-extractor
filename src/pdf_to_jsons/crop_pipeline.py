"""Extract MCQs with Docling layout, shrinking PDF crops, and portable local OCR."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any

from .crop_layout import Box, Segment, build_layout, page_boxes, question_segments
from .crop_ocr import create_crop_ocr, resolve_ocr_backend, parse_question
from .hybrid_mcq.run import save_json
from .pdf_to_mcqs import file_hash
from .run_converter import run_converter


class PDFRenderer:
    def __init__(self, path: Path, dpi: int):
        import pypdfium2
        self.pdf = pypdfium2.PdfDocument(str(path))
        self.scale = dpi / 72
        self.cache: dict[int, Any] = {}

    def close(self) -> None:
        for image in self.cache.values():
            image.close()
        self.pdf.close()

    def crop(self, segment: Segment) -> Any:
        region = segment.region
        if region.page not in self.cache:
            if not 1 <= region.page <= len(self.pdf):
                raise ValueError(f"Docling page {region.page} is outside the PDF")
            page = self.pdf[region.page - 1]
            try:
                width, height = page.get_size()
                if abs(width - region.page_width) > 2 or abs(height - region.page_height) > 2:
                    raise ValueError(f"Docling/PDF dimensions differ on page {region.page}")
                bitmap = page.render(scale=self.scale)
                try:
                    image = bitmap.to_pil().convert("RGB")
                finally:
                    bitmap.close()
            finally:
                page.close()
            if len(self.cache) >= 2:
                self.cache.pop(next(iter(self.cache))).close()
            self.cache[region.page] = image
        image = self.cache[region.page]
        sx, sy = image.width / region.page_width, image.height / region.page_height
        box = segment.box
        return image.crop((max(0, math.floor(box.left * sx)), max(0, math.floor(box.top * sy)),
                           min(image.width, math.ceil(box.right * sx)), min(image.height, math.ceil(box.bottom * sy))))


def stitch(images: list[Any]) -> Any:
    from PIL import Image
    width = max(image.width for image in images)
    canvas = Image.new("RGB", (width, sum(image.height for image in images) + 16 * (len(images) - 1)), "white")
    top = 0
    for image in images:
        canvas.paste(image, (0, top))
        top += image.height + 16
    return canvas


def render_patch(segments: list[Segment], renderer: Any) -> tuple[Any, list[int]]:
    images = []
    try:
        for segment in segments:
            images.append(renderer.crop(segment))
        return stitch(images), [image.height for image in images]
    finally:
        for image in images:
            image.close()


def content_floors(raw: dict, segments: list[Segment], padding: float) -> list[float]:
    """Protect all layout content, including diagrams OCR cannot read."""
    document = raw.get("document", raw)
    floors = []
    for segment in segments:
        boxes = page_boxes(raw, document, segment.region.page, segment.region.page_height)
        bottom = max((box.bottom for box in boxes
                      if box.left < segment.box.right and box.right > segment.box.left
                      and box.top < segment.box.bottom and box.bottom > segment.box.top),
                     default=segment.box.top)
        floors.append(min(segment.box.bottom, bottom + padding))
    return floors


def shrink_question(number: str, segments: list[Segment], renderer: Any, ocr: Any,
                    *, step: float, require_answer: bool, max_steps: int,
                    min_confidence: float, protected_bottoms: list[float],
                    padding: float = 4) -> tuple[dict, list[Segment], dict, list[dict]]:
    """OCR the full next-question boundary, trim empty tails, then verify final OCR.

    All OCR text and Docling layout content are protected. Merely retaining an
    answer marker is insufficient: it could discard the rest of the explanation.
    """
    if not segments or len(protected_bottoms) != len(segments):
        raise ValueError("Each crop segment must have a protected content boundary")
    full_image, pixel_heights = render_patch(segments, renderer)
    try:
        baseline = ocr(full_image, recognition="accurate")
    finally:
        full_image.close()
    trace = []

    def record(result: dict, selected: list[Segment], phase: str) -> None:
        parsed = parse_question(result["text"], number, require_answer=require_answer)
        trace.append({"iteration": len(trace) + 1, "phase": phase,
                      "segments": [part.to_dict() for part in selected],
                      "recognized_characters": len(result["text"]),
                      "option_labels": [option["label"] for option in parsed["options"]],
                      "has_answer": parsed["correct_answer"] is not None,
                      "has_solution": bool(parsed["solution"]),
                      "structurally_complete": not parsed["needs_review"]})

    record(baseline, segments, "full_boundary")
    selected = []
    offset = 0
    budget = max_steps
    for segment, pixel_height, protected in zip(segments, pixel_heights, protected_bottoms):
        floor = max(segment.box.top, protected)
        for line in baseline.get("lines", []):
            box = line["bbox"]
            if box[1] < offset + pixel_height and box[3] > offset:
                point_bottom = segment.box.top + min(pixel_height, box[3] - offset) / pixel_height * (segment.box.bottom - segment.box.top)
                floor = max(floor, point_bottom + padding)
        floor = min(segment.box.bottom, floor)
        # Calculate the equivalent small reverse steps without repeating OCR on
        # each empty strip. Geometry proves those strips contain no known ink.
        reductions = min(budget, max(0, math.floor((segment.box.bottom - floor) / step)))
        bottom = segment.box.bottom - reductions * step
        budget -= reductions
        if bottom > segment.box.top:
            selected.append(Segment(segment.region, Box(segment.box.left, segment.box.top,
                                                        segment.box.right, bottom)))
        offset += pixel_height + 16
    if not selected:
        selected = segments
    canvas, _ = render_patch(selected, renderer)
    try:
        final_ocr = ocr(canvas, recognition="accurate")
        record(final_ocr, selected, "final_shrunk_crop")
        # OCR near an image edge can change even when its boxes looked safe.
        # Fall back to the original crop if any baseline text would be lost.
        normalize = lambda text: " ".join(text.split())
        if selected != segments and normalize(final_ocr["text"]) != normalize(baseline["text"]):
            canvas.close()
            selected = segments
            canvas, _ = render_patch(selected, renderer)
            final_ocr = ocr(canvas, recognition="accurate")
            record(final_ocr, selected, "restored_full_boundary")
        parsed = parse_question(final_ocr["text"], number, require_answer=require_answer)
        if final_ocr["mean_confidence"] < min_confidence:
            parsed["warnings"].append("Final OCR confidence is below the configured threshold.")
        parsed["needs_review"] = bool(parsed["warnings"])
        return parsed, selected, {**final_ocr, "image": canvas}, trace
    except Exception:
        canvas.close()
        raise


def code_identity(backend: str = 'vision') -> str:
    digest = hashlib.sha256()
    for filename in ("crop_pipeline.py", "crop_layout.py", "crop_ocr.py", "run_converter.py"):
        digest.update(Path(__file__).with_name(filename).read_bytes())
    if backend == 'hf':
        native = Path(__file__).parent.parent / 'pdf_to_json_my'
        for filename in ('ocr.py', 'confidence.py'):
            digest.update((native / filename).read_bytes())
    return digest.hexdigest()


def run(args: argparse.Namespace) -> Path:
    pdf = args.pdf.expanduser().resolve()
    if not pdf.is_file() or pdf.suffix.lower() != ".pdf":
        raise ValueError(f"Expected an existing PDF: {pdf}")
    backend = resolve_ocr_backend(args.ocr_backend)
    # Initialize before expensive conversion to detect unavailable OCR early.
    print(f'Crop OCR: {backend} (device={args.ocr_device})', flush=True)
    ocr = create_crop_ocr(backend, languages=args.languages, device=args.ocr_device,
                          local_files_only=args.local_files_only)
    root = args.output.expanduser().resolve() / pdf.stem
    manifest_path = root / "crop_pipeline.json"
    settings = {key: getattr(args, key) for key in
                ("dpi", "step", "max_steps", "max_regions", "padding",
                 "min_confidence", "allow_missing_answer", "languages")}
    settings.update(ocr_backend=backend, ocr_device=args.ocr_device,
                    docling_ocr_engine='ocrmac' if backend == 'vision' else 'rapidocr')
    identity = {"pdf_sha256": file_hash(pdf), "code_sha256": code_identity(backend), "settings": settings}
    if args.resume:
        if not manifest_path.is_file():
            raise ValueError("No crop pipeline checkpoint found; run without --resume first")
        manifest = json.loads(manifest_path.read_text())
        if manifest["identity"] != identity:
            raise ValueError("PDF, code, or crop settings changed; choose a new output directory")
        raw_path = Path(manifest["raw_json"])
        if file_hash(raw_path) != manifest["raw_sha256"]:
            raise ValueError("Saved Docling JSON changed; choose a new output directory")
        if args.raw_json and args.raw_json.expanduser().resolve() != raw_path:
            raise ValueError("--raw-json differs from the checkpoint")
    else:
        if root.exists() and any(root.iterdir()):
            raise ValueError("Output directory is not empty; use --resume or another output directory")
        root.mkdir(parents=True, exist_ok=True)
        raw_path = args.raw_json.expanduser().resolve() if args.raw_json else root / "raw" / f"{pdf.stem}_raw.json"
        if not args.raw_json:
            run_converter(pdf, raw_path.parent, output_path=raw_path,
                          ocr_engine='ocrmac' if backend == 'vision' else 'rapidocr')
        manifest = {"identity": identity, "crop_strategy": "next_anchor_shrink", "raw_json": str(raw_path), "raw_sha256": file_hash(raw_path), "results": {}}
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    if "document" in raw and raw.get("status") != "success":
        raise ValueError("Docling conversion did not succeed; inspect its raw JSON")
    # Existing pipeline manifests provide a SHA-256 binding for imported raw JSON.
    original_manifest = raw_path.parent.parent / "pipeline.json"
    if original_manifest.is_file():
        expected = json.loads(original_manifest.read_text()).get("identity", {}).get("pdf_sha256")
        if expected and expected != identity["pdf_sha256"]:
            raise ValueError("Imported Docling JSON belongs to a different PDF")
    regions, anchors = build_layout(raw, padding=args.padding)
    if not anchors:
        raise ValueError("No question anchors found in Docling layout")
    document = raw.get("document", raw)
    manifest["regions"] = [asdict(region) for region in regions]
    manifest["anchors"] = [asdict(anchor) for anchor in anchors]
    records = manifest["results"]

    def checkpoint() -> None:
        save_json(manifest_path, manifest)
        questions = [record["question"] for record in records.values()]
        save_json(root / "mcqs.json", {
            "source_pdf": str(pdf), "source_json": str(raw_path),
            "extraction_method": f"docling_layout_adaptive_crop_{'apple_vision' if backend == 'vision' else 'hf_mobile'}",
            "question_count": len(questions), "anchor_count": len(anchors),
            "pending_count": len(anchors) - len(records), "questions": questions,
            "review_candidates": [], "llm_calls": 0,
            "limitations": ["OCR structure checks do not verify formulas or answer correctness.",
                            "Question anchors and column boundaries are inferred from layout; inspect saved crops."]})
    checkpoint()
    renderer = PDFRenderer(pdf, args.dpi)
    processed = 0
    try:
        for index, anchor in enumerate(anchors):
            key = f"{index:04d}_q{anchor.number}"
            if key in records:
                # Checkpointed artifacts must still match before skipping work.
                for relative, expected in records[key]["artifacts"].items():
                    path = root / relative
                    if not path.is_file() or file_hash(path) != expected:
                        raise ValueError(f"Saved artifact changed or is missing: {path}")
                continue
            if args.questions and anchor.number not in args.questions:
                continue
            if args.limit is not None and processed >= args.limit:
                break
            segments, truncated = question_segments(regions, anchors, index, padding=args.padding, max_regions=args.max_regions)
            if not segments:
                raise ValueError(f"No positive-area crop for question {anchor.number}")
            print(f"Question {anchor.number}: shrinking from next-question boundary across {len(segments)} column/page regions", flush=True)
            parsed, selected, final_ocr, trace = shrink_question(
                anchor.number, segments, renderer, ocr, step=args.step,
                require_answer=not args.allow_missing_answer, max_steps=args.max_steps,
                min_confidence=args.min_confidence, padding=args.padding,
                protected_bottoms=content_floors(raw, segments, args.padding))
            folder = root / "questions" / key
            folder.mkdir(parents=True, exist_ok=True)
            artifacts = {}
            patch = final_ocr.pop("image")
            try:
                patch.save(folder / "patch.png")
            finally:
                patch.close()
            # Save individual source segments as well as the stitched OCR patch.
            for part_index, segment in enumerate(selected):
                image = renderer.crop(segment)
                try:
                    image.save(folder / f"part_{part_index + 1:02d}.png")
                finally:
                    image.close()
            save_json(folder / "ocr.json", final_ocr)
            save_json(folder / "growth.json", {"strategy": "next_anchor_shrink", "iterations": trace})
            if truncated:
                parsed["warnings"].append("Question continuation exceeds --max-regions.")
            # Retain source evidence for math and diagrams; plain OCR cannot verify them.
            source_labels = {item.get("label") for bucket in ("texts", "pictures", "tables")
                             for item in document.get(bucket, []) for prov in item.get("prov", [])
                             for segment in selected if prov["page_no"] == segment.region.page
                             and (lambda box: box.left < segment.box.right and box.right > segment.box.left
                                  and box.top < segment.box.bottom and box.bottom > segment.box.top)(
                                      Box.from_docling(prov["bbox"], segment.region.page_height))}
            if source_labels & {"formula", "picture", "table"}:
                parsed["warnings"].append("Formula, diagram, or table is present; verify transcription against the saved patch.")
            parsed.update(needs_review=bool(parsed["warnings"]),
                          pages=list(dict.fromkeys(part.region.page for part in selected)),
                          source_refs=[anchor.source_ref], image_refs=[str((folder / "patch.png").relative_to(root))],
                          crops=[part.to_dict() for part in selected], ocr_confidence=final_ocr["mean_confidence"])
            save_json(folder / "question.json", parsed)
            for path in folder.iterdir():
                if path.is_file():
                    artifacts[str(path.relative_to(root))] = file_hash(path)
            records[key] = {"question": parsed, "artifacts": artifacts}
            checkpoint()
            processed += 1
            print(f"  {len(trace)} OCR checks; {len(parsed['options'])} options; needs_review={parsed['needs_review']}", flush=True)
    finally:
        renderer.close()
    print(f"Saved {len(records)}/{len(anchors)} question patches: {root / 'mcqs.json'}")
    return root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--raw-json", type=Path, help="Reuse existing Docling JSON instead of reconverting")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, help="Maximum new questions this run")
    parser.add_argument("--questions", nargs="+", help="Process selected question numbers only")
    parser.add_argument("--dpi", type=int, default=216)
    parser.add_argument("--initial-height", type=float, default=72, help=argparse.SUPPRESS)
    parser.add_argument("--step", type=float, default=24, help="Shrink increment in PDF points (empty strips only)")
    parser.add_argument("--padding", type=float, default=4)
    parser.add_argument("--max-steps", type=int, default=100, help="Maximum geometric shrink increments per question")
    parser.add_argument("--max-regions", type=int, default=4)
    parser.add_argument("--min-confidence", type=float, default=.75)
    parser.add_argument("--languages", nargs="+", default=["en-US"])
    parser.add_argument('--ocr-backend', choices=('auto', 'vision', 'hf'), default='auto',
                        help='Crop OCR: Apple Vision on macOS, native HF PP-OCRv5 elsewhere')
    parser.add_argument('--ocr-device', choices=('auto', 'cpu', 'cuda', 'mps'), default='auto')
    parser.add_argument('--local-files-only', action='store_true', help='Use cached native crop OCR weights only')
    parser.add_argument("--allow-missing-answer", action="store_true", help="Allow papers without printed answers/solutions")
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if any(not math.isfinite(value) or value <= 0 for value in (args.dpi, args.initial_height, args.step, args.max_steps, args.max_regions)):
        parser.error("DPI, crop heights, and iteration/region limits must be positive and finite")
    if not math.isfinite(args.padding) or args.padding < 0 or not 0 <= args.min_confidence <= 1 or (args.limit is not None and args.limit < 0):
        parser.error("Invalid padding, confidence threshold, or question limit")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)
    logging.basicConfig(level=logging.INFO)
    try:
        run(args)
    except (ImportError, OSError, RuntimeError, ValueError, KeyError) as exc:
        parser.exit(1, f"Crop pipeline failed: {exc}\n")


if __name__ == "__main__":
    main()
