#!/usr/bin/env python3
"""Convert a PDF with the local Docling checkout and save ConversionResult JSON."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from mcq_extractor.paths import REPOSITORY_ROOT
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from docling.datamodel.document import ConversionResult


def run_converter(
    pdf_path: Path, output_dir: Path, *, output_path: Path | None = None,
    ocr_engine: str | None = None,
) -> ConversionResult:
    """Return conv_res and save its public fields, including the document, as JSON."""
    pdf_path = pdf_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Import document_converter.py from this checkout rather than a site package.
    checkout = REPOSITORY_ROOT / "docling"
    if (checkout / "docling/document_converter.py").is_file():
        sys.path.insert(0, str(checkout))

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions()
    options.do_ocr = True
    if ocr_engine == "ocrmac":
        from docling.datamodel.pipeline_options import OcrMacOptions
        options.ocr_options = OcrMacOptions(lang=["en-US"], recognition="accurate", framework="vision")
    elif ocr_engine == 'rapidocr':
        from docling.datamodel.pipeline_options import RapidOcrOptions
        options.ocr_options = RapidOcrOptions(backend='torch', lang=['en'])
    elif ocr_engine is not None:
        raise ValueError(f'Unknown Docling OCR engine: {ocr_engine}')
    else:
        options.ocr_options.lang = ["en"]
    options.do_table_structure = True
    options.table_structure_options.do_cell_matching = True
    options.images_scale = 2.0
    options.generate_page_images = True
    options.generate_picture_images = True
    # Retain text cells and their from_ocr flags for inspection in the JSON.
    options.generate_parsed_pages = True

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )
    print(f"Converting {pdf_path}", flush=True)
    conv_res = converter.convert(pdf_path, raises_on_error=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        output_path.expanduser().resolve()
        if output_path is not None
        else output_dir / f"{pdf_path.stem}_conv_res.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Private runtime objects (backends, locks, image caches) are not serialized.
    output_path.write_text(conv_res.model_dump_json(indent=2), encoding="utf-8")

    print(f"Status: {conv_res.status.value}")
    print(f"Processed pages: {len(conv_res.pages)}")
    print(f"Saved conv_res: {output_path}")
    for error in conv_res.errors:
        print(f"Conversion error: {error.error_message}", file=sys.stderr)
    return conv_res


def main() -> None:
    workspace = REPOSITORY_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdf", type=Path, default=workspace / "dataset/pdfs/2021.pdf"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=workspace / "outputs/raw"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        conv_res = run_converter(args.pdf, args.output_dir)
    except ImportError as exc:
        parser.exit(
            1,
            f"Missing dependency in {sys.executable}: {exc}\n"
            'Install local Docling with: python3 -m pip install -e "./docling[standard]"\n',
        )
    except FileNotFoundError as exc:
        parser.exit(1, f"{exc}\n")

    from docling.datamodel.base_models import ConversionStatus

    if conv_res.status != ConversionStatus.SUCCESS:
        parser.exit(1, f"Conversion ended with status: {conv_res.status.value}\n")


if __name__ == "__main__":
    main()
