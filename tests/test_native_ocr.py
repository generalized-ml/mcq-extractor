import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mcq_extractor.ocr import cached_model, reading_order, rectify


class NativeOCRTests(unittest.TestCase):
    def test_complete_cache_never_requests_network(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ('model.safetensors', 'config.json', 'preprocessor_config.json'):
                (Path(directory) / name).touch()
            with patch('huggingface_hub.snapshot_download', return_value=directory) as download:
                self.assertEqual(cached_model('org/repo', 'commit'), directory)
                download.assert_called_once()
                self.assertTrue(download.call_args.kwargs['local_files_only'])

    def test_partial_cache_offline_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('huggingface_hub.snapshot_download', return_value=directory) as download:
                with self.assertRaisesRegex(RuntimeError, 'not fully cached'):
                    cached_model('org/repo', 'commit', local_files_only=True)
                self.assertEqual(download.call_count, 1)

    def test_local_finetuned_directory_skips_hub(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('huggingface_hub.snapshot_download') as download:
                self.assertEqual(cached_model(directory, 'unused'), str(Path(directory).resolve()))
                download.assert_not_called()

    def test_box_reading_order(self):
        left = [[0, 12], [20, 12], [20, 30], [0, 30]]
        right = [[40, 10], [60, 10], [60, 30], [40, 30]]
        below = [[0, 60], [20, 60], [20, 80], [0, 80]]
        self.assertEqual(reading_order([below, right, left]), [left, right, below])

    def test_rectification_keeps_horizontal_orientation(self):
        import numpy as np
        source = np.zeros((20, 50, 3), dtype=np.uint8)
        source[:, 25:] = 255
        crop = rectify(source, [[0, 0], [49, 0], [49, 19], [0, 19]])
        pixels = np.asarray(crop)
        self.assertGreater(crop.width, crop.height)
        self.assertLess(pixels[:, 0].mean(), pixels[:, -1].mean())
        crop.close()
