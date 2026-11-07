import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.paddle.mobile import MobileReader
from experiments.paddle.pipeline import build_parser
from experiments.paddle.structure import route_question


class MobileTests(unittest.TestCase):
    def reader(self, results):
        self.settings = {}
        owner = self
        class OCR:
            def __init__(self, **kwargs):
                owner.settings.update(kwargs)
            def predict(self, path):
                return results
        with patch.dict(sys.modules, {'paddleocr': types.SimpleNamespace(PaddleOCR=OCR)}):
            return MobileReader()

    def test_mobile_default_and_no_layout_preprocessing(self):
        self.assertEqual(build_parser().parse_args(['2021.pdf']).ocr_backend, 'mobile')
        reader = self.reader([{'rec_texts': ['Question', '(1) A'], 'rec_scores': [.99, .2]}])
        part = reader.read(Path('crop.png'))
        self.assertFalse(self.settings['use_doc_orientation_classify'])
        self.assertFalse(self.settings['use_doc_unwarping'])
        self.assertFalse(self.settings['use_textline_orientation'])
        self.assertEqual(self.settings['text_rec_score_thresh'], 0)
        self.assertEqual(part['text'], 'Question\n(1) A')
        self.assertEqual(part['confidence']['low_token_fraction'], .5)
        self.assertTrue(route_question({'warnings': []}, [part], threshold=.85,
                                      max_low_token_fraction=.1, crop_warnings=[])['review_required'])

    def test_empty_ocr_is_not_high_confidence(self):
        part = self.reader([{'rec_texts': [], 'rec_scores': []}]).read(Path('crop.png'))
        self.assertIsNone(part['confidence']['score'])

    def test_mismatched_scores_fail(self):
        reader = self.reader([{'rec_texts': ['text'], 'rec_scores': []}])
        with self.assertRaises(ValueError):
            reader.read(Path('crop.png'))

    def test_unsupported_options_fail_before_loading(self):
        for kwargs in ({'device': 'mps'}, {'local_files_only': True}):
            with self.assertRaises(ValueError):
                MobileReader(**kwargs)

class ConfidenceScopeTests(unittest.TestCase):
    def part(self, lines):
        return {'text': '\n'.join(t for t, s in lines),
                'lines': [{'text': t, 'score': s} for t, s in lines]}

    def test_excludes_late_solution_but_preserves_text(self):
        from experiments.paddle.mobile import scope_confidence
        p = self.part([('Question and options', .99), ('Solution: one two three', .98),
                       ('Badly read long explanation', .1)])
        original = p['text']
        scope_confidence([p])
        self.assertEqual(p['text'], original)
        self.assertEqual(p['confidence']['scored_lines'], 2)
        self.assertGreater(p['confidence']['score'], .97)
        self.assertEqual(p['confidence_scope']['text'], 'Question and options\nSolution: one two three')

    def test_cross_crop_three_words_and_excluded_continuation(self):
        from experiments.paddle.mobile import scope_confidence
        parts = [self.part([('Q ANSWER: (2)', .99)]),
                 self.part([('because gravity', .98), ('later text', .1)]),
                 self.part([('more explanation', .1)])]
        selected = scope_confidence(parts)
        self.assertEqual(len(selected), 2)
        self.assertEqual(parts[1]['confidence']['scored_lines'], 1)
        self.assertTrue(parts[2]['confidence_scope']['excluded'])

    def test_no_marker_full_text_and_partial_boundary_line(self):
        from experiments.paddle.mobile import scope_confidence
        p = self.part([('question', .99), ('option', .1)])
        scope_confidence([p])
        self.assertEqual(p['confidence']['scored_lines'], 2)
        self.assertFalse(p['confidence_scope']['marker_found'])
        p = self.part([('Answer: one two three four five', .8)])
        scope_confidence([p])
        self.assertEqual(p['confidence_scope']['text'], 'Answer: one two three')
        self.assertAlmostEqual(p['confidence']['score'], .8)

    def test_short_tail_and_word_boundary(self):
        from experiments.paddle.mobile import scope_confidence
        p = self.part([('dissolution answers', .9), ('Solution: (2)', .9)])
        scope_confidence([p])
        self.assertEqual(p['confidence_scope']['text'], p['text'])
