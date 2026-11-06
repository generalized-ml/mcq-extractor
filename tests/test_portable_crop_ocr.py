"""Cross-platform crop OCR contracts and conversion configuration, offline."""

import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from pdf_to_jsons.crop_ocr import HFMobileCropOCR, create_crop_ocr, format_lines, resolve_ocr_backend
from pdf_to_jsons.crop_pipeline import build_parser as crop_parser, run as crop_run
from pdf_to_jsons.run_converter import run_converter
from pdf_to_json_my.pipeline import build_parser as native_parser
from pdf_to_json_my.crops import prepare_crops


class PortableCropTests(unittest.TestCase):
    def test_auto_selects_vision_only_on_macos(self):
        self.assertEqual(resolve_ocr_backend('auto', platform='darwin'), 'vision')
        for platform in ('linux', 'win32'):
            self.assertEqual(resolve_ocr_backend('auto', platform=platform), 'hf')
        self.assertEqual(resolve_ocr_backend('hf', platform='darwin'), 'hf')

    def test_hf_factory_never_initializes_vision(self):
        with patch('pdf_to_jsons.crop_ocr.MacVisionOCR', side_effect=AssertionError('No Apple OCR')), \
             patch('pdf_to_jsons.crop_ocr.HFMobileCropOCR') as factory:
            create_crop_ocr('hf', device='cpu', local_files_only=True)
        factory.assert_called_once_with(device='cpu', local_files_only=True)

    def test_native_adapter_returns_pixel_bounds_and_preserves_same_row_fragments(self):
        reader = Mock(device='cuda')
        reader.read_image.return_value = {'lines': [
            {'text': '(2) Two', 'score': .8, 'box': [[100, 20], [160, 20], [160, 40], [100, 40]]},
            {'text': '(1) One', 'score': 1., 'box': [[10, 20], [80, 20], [80, 40], [10, 40]]},
            {'text': 'Answer (1)', 'score': .9, 'box': [[10, 80], [90, 80], [90, 100], [10, 100]]},
        ]}
        with patch('pdf_to_json_my.ocr.HFMobileReader', return_value=reader):
            backend = HFMobileCropOCR(device='cuda')
        with Image.new('RGB', (200, 120), 'white') as image:
            result = backend(image)
            reader.read_image.assert_called_once_with(image)
            self.assertEqual(image.size, (200, 120))
        self.assertEqual(result['lines'][0]['bbox'], [100, 20, 160, 40])
        self.assertEqual(result['text'], '(1) One  (2) Two\nAnswer (1)')
        self.assertAlmostEqual(result['mean_confidence'], .9)
        self.assertEqual(result['backend'], 'hf')

    def test_empty_ocr_preserves_safe_confidence_contract(self):
        self.assertEqual(format_lines([])['mean_confidence'], 0)
        self.assertEqual(format_lines([])['text'], '')

    def test_crop_pipeline_selects_portable_docling_conversion_before_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf = root / '2023.pdf'
            pdf.write_bytes(b'fixture')
            args = crop_parser().parse_args([str(pdf), str(root / 'out'), '--ocr-backend', 'hf', '--ocr-device', 'cpu'])
            def convert(pdf, folder, *, output_path, ocr_engine):
                self.assertEqual(ocr_engine, 'rapidocr')
                output_path.parent.mkdir(parents=True)
                output_path.write_text(json.dumps({'status': 'success', 'document': {}}))
            with patch('pdf_to_jsons.crop_pipeline.create_crop_ocr'), \
                 patch('pdf_to_jsons.crop_pipeline.run_converter', side_effect=convert) as conversion, \
                 patch('pdf_to_jsons.crop_pipeline.build_layout', side_effect=RuntimeError('layout reached')), \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, 'layout reached'):
                    crop_run(args)
            conversion.assert_called_once()

    def test_docling_uses_rapidocr_torch_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf = root / 'paper.pdf'
            pdf.write_bytes(b'fixture')
            result = Mock(pages={}, errors=[], status=SimpleNamespace(value='success'))
            result.model_dump_json.return_value = '{}'
            with patch('docling.document_converter.DocumentConverter') as converter, \
                 contextlib.redirect_stdout(io.StringIO()):
                converter.return_value.convert.return_value = result
                run_converter(pdf, root / 'out', ocr_engine='rapidocr')
            options = next(iter(converter.call_args.kwargs['format_options'].values())).pipeline_options
            self.assertEqual(options.ocr_options.backend, 'torch')
            self.assertEqual(options.ocr_options.lang, ['en'])

    def test_native_cli_forwards_crop_options(self):
        args = native_parser().parse_args(['paper.pdf', '--crop-ocr', 'hf', '--crop-device', 'cpu', '--local-files-only'])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch('pdf_to_json_my.crops.run_crops', side_effect=RuntimeError('forwarded')) as run:
                with self.assertRaisesRegex(RuntimeError, 'forwarded'):
                    prepare_crops(Path('/paper.pdf'), root, args)
            forwarded = run.call_args.args[0]
            self.assertEqual(forwarded.ocr_backend, 'hf')
            self.assertEqual(forwarded.ocr_device, 'cpu')
            self.assertTrue(forwarded.local_files_only)
