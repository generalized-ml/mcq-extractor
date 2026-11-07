"""Folder discovery and pairing only; no model loading or evaluation execution."""

import json
from pathlib import Path
import tempfile
import unittest

from mcq_extractor.evaluation.evaluate import golden_file, load_combined, prediction_files


class FolderInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = self.root / 'any_experiment_name'
        self.inputs.mkdir()
        self.golden = self.root / 'references'
        self.golden.mkdir()
        self.record = {'question_number': '1', 'question_text': 'Fixture',
                       'options': ['a', 'b', 'c', 'd'], 'correct_answer': '1'}

    def write(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'questions': [self.record], 'llm_calls': []}))

    def test_discovers_only_present_pdfs_with_arbitrary_names(self):
        self.write(self.inputs / 'paperA' / 'final_mcqs.json')
        self.write(self.golden / 'paperA' / 'final_mcqs.json')
        actual, golden, _, _, _, coverage = load_combined(self.inputs, 'custom-label', self.golden)
        self.assertEqual([q['identity'] for q in actual], ['paperA:1'])
        self.assertEqual([q['identity'] for q in golden], ['paperA:1'])
        self.assertEqual(coverage, [('paperA', 1, 1, True)])

    def test_legacy_format_does_not_depend_on_experiment_label(self):
        legacy = self.inputs / '2021' / 'mcq_with_llm' / '2021_mcqs_with_llm.json'
        self.write(legacy)
        self.write(self.golden / 'neet_2021' / '2021_mcqs.json')
        actual, _, _, _, _, _ = load_combined(self.inputs, 'renamed', self.golden)
        self.assertEqual(actual[0]['identity'], '2021:1')
        canonical = self.inputs / '2021' / 'final_mcqs.json'
        self.write(canonical)
        self.assertEqual(prediction_files(self.inputs), {'2021': canonical})

    def test_single_pdf_folder_and_single_golden_folder(self):
        paper = self.inputs / '2023'
        reference = self.golden / 'neet_2023'
        self.write(paper / 'final_mcqs.json')
        self.write(reference / '2023_mcqs.json')
        actual, _, _, _, _, coverage = load_combined(paper, 'custom', reference)
        self.assertEqual(len(actual), 1)
        self.assertEqual(coverage, [(2023, 1, 1, True)])

    def test_explicit_years_still_include_missing_outputs(self):
        self.write(self.inputs / '2021' / 'final_mcqs.json')
        for year in (2021, 2022):
            self.write(self.golden / f'neet_{year}' / f'{year}_mcqs.json')
        actual, golden, cost, _, _, coverage = load_combined(self.inputs, 'custom', self.golden, [2021, 2022])
        self.assertEqual((len(actual), len(golden)), (1, 2))
        self.assertEqual(coverage[-1], (2022, 0, 1, False))

    def test_flat_golden_folder(self):
        path = self.golden / 'paperA.json'
        self.write(path)
        self.assertEqual(golden_file(self.golden, 'paperA'), path)

    def test_renamed_single_pdf_folder_uses_recorded_source_pdf(self):
        (self.inputs / 'final_mcqs.json').write_text(json.dumps({
            'source_pdf': '/original/location/2023.pdf', 'questions': [self.record]}))
        self.write(self.golden / 'neet_2023' / '2023_mcqs.json')
        actual, _, _, _, _, _ = load_combined(self.inputs, 'custom', self.golden)
        self.assertEqual(actual[0]['identity'], '2023:1')

    def test_missing_reference_and_empty_experiment_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, 'No final_mcqs.json'):
            load_combined(self.inputs, 'custom', self.golden)
        self.write(self.inputs / '2021' / 'final_mcqs.json')
        with self.assertRaisesRegex(ValueError, 'Missing golden JSON for 2021'):
            load_combined(self.inputs, 'custom', self.golden)
