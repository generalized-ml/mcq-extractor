"""Offline checks for the combined PDF/MCQ/LLM pipeline."""

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pdf_to_jsons import pdf_to_mcqs as pipeline


def raw_fixture():
    rows = [("1.", "Pick a number."), ("(1)", "One"), ("(2)", "Two")]
    texts = [{"self_ref": f"#/texts/{i}", "label": "list_item", "marker": marker,
              "text": text, "children": [], "prov": [{"page_no": 1}]}
             for i, (marker, text) in enumerate(rows)]
    return {"status": "success", "document": {
        "name": "paper", "texts": texts,
        "body": {"children": [{"cref": t["self_ref"]} for t in texts]},
        "pages": {"1": {"image": {"uri": "data:image/png;base64,dGVzdA=="}}},
    }}


def llm_fixture():
    return {"extraction": {"question_number": "1", "question_text": "Pick $10^{5}$.",
            "options": [{"label": "1", "text": "One", "image_refs": []},
                        {"label": "2", "text": "Two", "image_refs": []}],
            "correct_answer": None, "image_refs": [], "status": "complete", "warnings": []},
            "usage": {}, "response_id": "offline", "model": "test/vision"}


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        pdf = root / "paper.pdf"
        pdf.write_bytes(b"offline PDF placeholder")
        self.args = argparse.Namespace(pdf_path=pdf, output_folder=root / "out",
            model="test/vision", limit=None, max_images=8, retries=0,
            timeout=1, resume=False, dry_run=False, review_all=False)

    @staticmethod
    def convert(pdf_path, output_dir, *, output_path):
        pipeline.save_json(output_path, raw_fixture())

    def test_output_layout_and_resume_without_repeated_work(self):
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "offline"}), \
             patch.object(pipeline, "run_converter", side_effect=self.convert) as convert, \
             patch("pdf_to_jsons.hybrid_mcq.run.call_openrouter", return_value=llm_fixture()) as llm, \
             contextlib.redirect_stdout(io.StringIO()):
            root = pipeline.run_pipeline(self.args)
            self.args.resume = True
            pipeline.run_pipeline(self.args)
        self.assertEqual(convert.call_count, 1)
        self.assertEqual(llm.call_count, 1)
        raw = root / "raw/paper_raw.json"
        mcqs = root / "mcqs/paper_mcqs.json"
        final = root / "mcq_with_llm/paper_mcqs_with_llm.json"
        self.assertTrue(raw.is_file())
        self.assertTrue(mcqs.is_file())
        data = json.loads(final.read_text())
        self.assertEqual(json.loads((root / 'final_mcqs.json').read_text()), data)
        self.assertEqual(json.loads((root / 'mcqs.json').read_text()), json.loads(mcqs.read_text()))
        self.assertEqual(data["source_json"], str(raw))
        self.assertEqual(data["questions"][0]["question_text"], "Pick $10^{5}$.")

    def test_changed_pdf_rejected_before_reconversion(self):
        self.args.dry_run = True
        with patch.object(pipeline, "run_converter", side_effect=self.convert) as convert, \
             contextlib.redirect_stdout(io.StringIO()):
            pipeline.run_pipeline(self.args)
            self.args.pdf_path.write_bytes(b"changed PDF")
            self.args.resume = True
            with self.assertRaisesRegex(ValueError, "PDF or conversion/extraction code changed"):
                pipeline.run_pipeline(self.args)
        self.assertEqual(convert.call_count, 1)

    def test_failed_llm_retains_output_and_signals_retry(self):
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "offline"}), \
             patch.object(pipeline, "run_converter", side_effect=self.convert), \
             patch("pdf_to_jsons.hybrid_mcq.run.call_openrouter", side_effect=RuntimeError("offline error")), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "use --resume"):
                pipeline.run_pipeline(self.args)
        final = self.args.output_folder / "paper/mcq_with_llm/paper_mcqs_with_llm.json"
        data = json.loads(final.read_text())
        self.assertEqual(json.loads((final.parent.parent / 'final_mcqs.json').read_text()), data)
        self.assertEqual(data["questions"][0]["llm_error"], "offline error")
        self.assertTrue(data["questions"][0]["needs_review"])

    def test_existing_output_is_not_overwritten(self):
        self.args.dry_run = True
        with patch.object(pipeline, "run_converter", side_effect=self.convert) as convert, \
             contextlib.redirect_stdout(io.StringIO()):
            pipeline.run_pipeline(self.args)
            with self.assertRaisesRegex(ValueError, "already exists"):
                pipeline.run_pipeline(self.args)
        self.assertEqual(convert.call_count, 1)


if __name__ == "__main__":
    unittest.main()
