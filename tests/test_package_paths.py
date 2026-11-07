"""Installed packages must never use site-packages as an output workspace."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mcq_extractor.paths import workspace_root


class PackageWorkspaceTests(unittest.TestCase):
    def test_installed_package_uses_working_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            installed = root / 'lib/python3.11/site-packages/mcq_extractor/paths.py'
            with patch.dict('os.environ', {}, clear=True), \
                 patch('mcq_extractor.paths.__file__', str(installed)), \
                 patch('mcq_extractor.paths.Path.cwd', return_value=root):
                self.assertEqual(workspace_root(), root.resolve())

    def test_workspace_override_takes_precedence(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict('os.environ', {'MCQ_WORKSPACE': temporary}):
                self.assertEqual(workspace_root(), Path(temporary).resolve())

    def test_source_checkout_uses_repository_root(self):
        root = Path(__file__).resolve().parents[1]
        with patch.dict('os.environ', {}, clear=True):
            self.assertEqual(workspace_root(), root)


if __name__ == '__main__':
    unittest.main()
