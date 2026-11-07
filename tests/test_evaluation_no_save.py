"""Verify CLI write behavior with scoring/model loading completely mocked."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mcq_extractor.evaluation.evaluate import main


class PrintOnlyTests(unittest.TestCase):
    def test_no_save_preserves_csv_and_does_not_create_output_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root / 'experiment' / '2023'
            references = root / 'golden'
            inputs.mkdir(parents=True)
            references.mkdir()
            fixture = {'questions': [{'question_number': '1', 'question_text': 'Fixture',
                                     'options': ['a', 'b', 'c', 'd']}]}
            prediction = inputs / 'final_mcqs.json'
            prediction.write_text(json.dumps(fixture))
            golden = references / '2023_mcqs.json'
            golden.write_text(json.dumps(fixture))
            summary = {'keyword_score': .5, 'semantic_score': .8, 'aggregate_score': .65,
                       'solution_golden_empty_count': 0}
            for existing in (False, True):
                output = root / ('existing' if existing else 'not_created')
                if existing:
                    output.mkdir()
                    (output / 'experiment.csv').write_bytes(b'existing CSV must stay identical\n')
                for command in ([str(inputs.parent), str(references)],
                                ['--mcq-json', str(prediction), '--golden-json', str(golden)]):
                    printed = io.StringIO()
                    with patch('sys.argv', ['evaluate', *command, '--no-save', '--output-dir', str(output)]), \
                         patch('mcq_extractor.evaluation.evaluate.SentenceEncoder'), \
                         patch('mcq_extractor.evaluation.evaluate.evaluate', return_value=([], summary.copy())), \
                         patch('mcq_extractor.evaluation.evaluate.save_results') as save, \
                         contextlib.redirect_stdout(printed):
                        main()
                    save.assert_not_called()
                    self.assertIn('keyword_score: 0.50000000', printed.getvalue())
                    self.assertIn('CSV not updated', printed.getvalue())
                    if existing:
                        self.assertEqual((output / 'experiment.csv').read_bytes(),
                                         b'existing CSV must stay identical\n')
                    else:
                        self.assertFalse(output.exists())
