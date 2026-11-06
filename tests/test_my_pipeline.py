"""Paddle parsing, probability routing, text-only review, and checkpoint regressions."""

import contextlib
import copy
import io
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from pdf_to_json_my.crops import prepare_crops
from pdf_to_json_paddle.ocr import token_confidence
from pdf_to_json_my.pipeline import build_parser, run
from pdf_to_json_my.review import review_question
from pdf_to_json_my.structure import parse_mcq, route_question
from pdf_to_jsons.pdf_to_mcqs import file_hash


class SpawnReader:
    """Importable fake reader for real spawn-process regression checks."""
    def __init__(self, **kwargs):
        self.model_info = {'fake': 'spawn-regression'}

    def read(self, path):
        return transcription(path.parent.name.split('_q')[1])


def text(number="1"):
    return f"{number}. Find $x^{{2}}$.\n(1) One\n(2) Two\n(3) Three\n(4) Four\nAnswer: (2)\nSol. Printed explanation."


def transcription(number="1", score=.95, truncated=False):
    return {"text": text(number), "lines": [{"text": t, "score": score} for t in text(number).split("\n")], "confidence": {"score": score, "low_token_fraction": .01},
            "truncated": truncated, "model": "test/paddle", "revision": "test"}


class StructureTests(unittest.TestCase):
    def test_math_and_solution_preserved(self):
        question = parse_mcq(text(), "1")
        self.assertEqual(question["question_text"], "Find $x^{2}$.")
        self.assertEqual(question["correct_answer"], "2")
        self.assertEqual(question["solution"], "Printed explanation.")
        self.assertFalse(question["needs_review"])

    def test_matching_table_labels_are_not_option_boundaries(self):
        value = "1. Match\n(A) Drift\n(B) Resistivity\n(C) Time\n(D) Current\n" + text().split("\n", 1)[1]
        question = parse_mcq(value, "1")
        self.assertIn("(D) Current", question["question_text"])
        self.assertEqual(len(question["options"]), 4)

    def test_parentheses_in_latex_do_not_become_options(self):
        question = parse_mcq(text().replace("One", r"$f(x)=(1)+(2)$"), "1")
        self.assertEqual(question["options"][0]["text"], r"$f(x)=(1)+(2)$")
        self.assertEqual(len(question["options"]), 4)

    def test_letter_labels_and_inline_options(self):
        value = "1. Pick\n(A) a (B) b (C) c (D) d\nAnswer: B"
        # Inline options are accepted only when the complete ordered sequence is clear.
        for source in (value, "1. Pick (A) a (B) b (C) c (D) d\nAnswer: B"):
            question = parse_mcq(source, "1")
            self.assertEqual([q["label"] for q in question["options"]], ["1", "2", "3", "4"])
            self.assertEqual(question["correct_answer"], "2")

    def test_high_confidence_requires_structure_and_complete_generation(self):
        question = parse_mcq(text(), "1")
        route = lambda q, p: route_question(q, p, threshold=.85, max_low_token_fraction=.1)
        self.assertFalse(route(question, [transcription()])["review_required"])
        self.assertTrue(route(question, [transcription(score=.5)])["review_required"])
        self.assertTrue(route(question, [transcription(truncated=True)])["review_required"])
        self.assertTrue(route(parse_mcq("1. Missing options", "1"), [transcription()])["review_required"])
        self.assertTrue(route(question, [transcription(score=None)])["review_required"])

    def test_lowest_crop_score_and_uncertain_tokens_route_review(self):
        question = parse_mcq(text(), "1")
        result = route_question(question, [transcription(), transcription(score=.6)], threshold=.85, max_low_token_fraction=.1)
        self.assertEqual(result["score"], .6)
        self.assertTrue(result["review_required"])
        uncertain = transcription()
        uncertain["confidence"]["low_token_fraction"] = .2
        self.assertTrue(route_question(question, [uncertain], threshold=.85, max_low_token_fraction=.1)["review_required"])

    def test_token_confidence_excludes_special_tokens_and_handles_unknown(self):
        result = token_confidence([(1, math.log(.9)), (2, math.log(.4)), (9, -100)], {9})
        self.assertAlmostEqual(result["score"], .6)
        self.assertEqual(result["low_token_fraction"], .5)
        self.assertIsNone(token_confidence([], set())["score"])
        self.assertIsNone(token_confidence([(1, float("nan"))], set())["score"])


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pdf = self.root / "paper.pdf"
        self.pdf.write_bytes(b"source PDF fixture")
        self.source = self.root / "source"
        records = {}
        for index in range(2):
            key = f"{index:04d}_q{index + 1}"
            folder = self.source / "questions" / key
            folder.mkdir(parents=True)
            for name in ("patch.png", "part_01.png"):
                Image.new("RGB", (50, 50), "white").save(folder / name)
            records[key] = {"question": {"question_number": str(index + 1), "warnings": [],
                                         "image_refs": [str((folder / "patch.png").relative_to(self.source))]},
                            "artifacts": {str(p.relative_to(self.source)): file_hash(p) for p in folder.iterdir()}}
        (self.source / "crop_pipeline.json").write_text(json.dumps({"identity": {"pdf_sha256": file_hash(self.pdf)},
                                                                   "results": records, "anchors": [{}, {}]}))
        self.args = build_parser().parse_args([str(self.pdf), "--output", str(self.root / "out"),
                                               "--crops-dir", str(self.source), "--workers", "1", "--device", "cpu"])
        self.ocr_calls, self.review_calls = [], []
        owner = self

        class Reader:
            def __init__(self, *args, **kwargs):
                pass

            def read(self, path):
                number = path.parent.name.split("_q")[1]
                owner.ocr_calls.append(number)
                return transcription(number, score=.95 if number == "1" else .5)
        self.reader = Reader

    def review(self, draft, source_text, **kwargs):
        self.review_calls.append(draft["question_number"])
        extraction = {key: copy.deepcopy(draft[key]) for key in
                      ("question_number", "question_text", "options", "correct_answer", "solution", "image_refs")}
        extraction.update(status="complete", warnings=[])
        kwargs["record_attempt"]({"response_id": "response-2", "status": "success", "usage": {"cost": .001}})
        return {"extraction": extraction, "model": "test/reviewer", "usage": {"cost": .001}, "response_id": "response-2"}

    def execute(self, reviewer=None):
        with contextlib.redirect_stdout(io.StringIO()), patch.dict(os.environ, {"OPENROUTER_API_KEY": "offline"}):
            return run(self.args, reader_factory=self.reader, reviewer=reviewer or self.review)

    def test_only_low_confidence_is_sent_and_resume_reuses_every_stage(self):
        path = self.execute()
        data = json.loads(path.read_text())
        self.assertEqual(self.review_calls, ["2"])
        self.assertEqual(data["llm_selected_count"], 1)
        self.assertEqual(data["questions"][0]["review_method"], "not_needed")
        self.assertTrue(data["questions"][1]["needs_review"])
        self.assertEqual(len(data["llm_calls"]), 1)
        self.args.resume = True
        self.execute()
        self.assertEqual(self.ocr_calls, ["1", "2"])
        self.assertEqual(self.review_calls, ["2"])

    def test_ocr_only_then_resume_runs_selected_review_without_retranscribing(self):
        self.args.ocr_only = True
        path = self.execute()
        self.assertEqual(self.review_calls, [])
        self.assertEqual(json.loads(path.read_text())["llm_pending_count"], 1)
        self.args.ocr_only = False
        self.args.resume = True
        self.execute()
        self.assertEqual(self.ocr_calls, ["1", "2"])
        self.assertEqual(self.review_calls, ["2"])

    def test_changed_crop_or_threshold_is_rejected(self):
        path = self.execute()
        self.args.resume = True
        self.args.confidence_threshold = .99
        with self.assertRaisesRegex(ValueError, "settings changed"):
            self.execute()
        self.args.confidence_threshold = .85
        (path.parent / "questions/0000_q1/part_01.png").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Saved crop changed"):
            self.execute()

    def test_failed_review_can_resume_without_losing_attempt_costs(self):
        def failing(draft, source_text, **kwargs):
            kwargs["record_attempt"]({"status": "error", "usage": {"cost": .002}, "response_id": "bad"})
            raise RuntimeError("offline provider failure")
        with self.assertRaisesRegex(RuntimeError, "Some LLM reviews failed"):
            self.execute(reviewer=failing)
        self.args.resume = True
        path = self.execute()
        data = json.loads(path.read_text())
        self.assertEqual(len(data["llm_calls"]), 2)
        self.assertEqual(self.ocr_calls, ["1", "2"])

    def test_crop_pdf_identity_must_match(self):
        self.pdf.write_bytes(b"different PDF")
        with self.assertRaisesRegex(ValueError, "different PDF"):
            self.execute()

    def test_parallel_spawn_preserves_question_order_and_resumes(self):
        self.reader = SpawnReader
        self.args.workers = 2
        self.args.physical_cores = 4
        self.args.ocr_only = True
        path = self.execute()
        data = json.loads(path.read_text())
        self.assertEqual([q['question_number'] for q in data['questions']], ['1', '2'])
        state = json.loads((path.parent / 'pipeline.json').read_text())
        self.assertEqual(state['ocr_execution']['workers'], 2)
        self.assertTrue(all(r['ocr_execution']['worker_pid'] != os.getpid() for r in state['results'].values()))
        self.args.resume = True
        self.args.workers = 1
        with patch('pdf_to_json_my.question_workers.HFMobileReader', side_effect=AssertionError('No pending OCR')):
            self.execute()

    def test_limit_caps_submitted_questions_before_worker_creation(self):
        self.args.workers = 4
        self.args.limit = 1
        self.args.ocr_only = True
        path = self.execute()
        self.assertEqual(self.ocr_calls, ['1'])
        self.assertEqual(json.loads(path.read_text())['pending_ocr_count'], 1)
        state = json.loads((path.parent / 'pipeline.json').read_text())
        self.assertEqual(state['ocr_execution']['workers'], 1)

    def test_completed_question_is_checkpointed_before_worker_failure(self):
        def completed_then_failed(items, root, options, settings, **kwargs):
            yield {'key': items[1]['key'], 'transcripts': [transcription('2')],
                   'models': {}, 'execution': {'seconds': .1, 'worker_pid': 123}}
            raise RuntimeError('Worker failed')

        self.args.ocr_only = True
        with patch('pdf_to_json_my.pipeline.transcribe_questions', side_effect=completed_then_failed):
            with self.assertRaisesRegex(RuntimeError, 'Worker failed'):
                self.execute()
        self.args.resume = True
        path = self.execute()
        self.assertEqual(self.ocr_calls, ['1'])
        self.assertEqual([q['question_number'] for q in json.loads(path.read_text())['questions']], ['1', '2'])


class ReviewTests(unittest.TestCase):
    def test_request_is_text_only_and_includes_full_ocr(self):
        draft = parse_mcq(text(), "1")
        extraction = {key: draft[key] for key in ("question_number", "question_text", "options", "correct_answer", "solution", "image_refs")}
        extraction.update(status="complete", warnings=[])
        payload = {"id": "response", "usage": {"cost": .01}, "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(extraction)}}]}
        captured, attempts = [], []

        def request(req, **kwargs):
            captured.append(json.loads(req.data))
            return io.StringIO(json.dumps(payload))

        with patch("pdf_to_jsons.json_crops.urlopen", side_effect=request):
            review_question(draft, text(), model="test", api_key="offline", timeout=1, retries=0, record_attempt=attempts.append)
        content = captured[0]["messages"][1]["content"]
        self.assertTrue(all(item["type"] == "text" for item in content))
        self.assertEqual(json.loads(content[0]["text"])["source_ocr_text"], text())
        self.assertEqual(attempts[0]["usage"]["cost"], .01)


if __name__ == "__main__":
    unittest.main()
