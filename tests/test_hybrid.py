"""Offline regression checks for evidence selection and LLM result handling."""

import argparse
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.docling import review as hybrid
from mcq_extractor import review_api


def question(number="1", review=True):
    return {"question_number": number, "question_text": "Draft", "options": [],
            "correct_answer": None, "needs_review": review, "pages": [1],
            "source_refs": ["#/texts/0"], "picture_refs": ["#/pictures/0"],
            "table_refs": ["#/tables/0"], "warnings": ["Missing options"]}


def document():
    uri = "data:image/png;base64,dGVzdA=="
    return {"texts": [{"text": "Source stem", "children": []},
                      {"text": "Picture child", "children": []}],
            "pictures": [{"image": {"uri": uri},
                          "children": [{"cref": "#/texts/1"}]}],
            "tables": [{"data": {"table_cells": [{"text": "Table option"}]}}],
            "pages": {"1": {"image": {"uri": uri}}}}


def extraction():
    return {"question_number": "1", "question_text": "Recovered stem",
            "options": [{"label": "A", "text": "First", "image_refs": []},
                        {"label": "B", "text": "Second", "image_refs": []}],
            "correct_answer": "B", "image_refs": [], "status": "complete",
            "warnings": []}


class HybridTests(unittest.TestCase):
    def test_evidence_includes_page_picture_children_and_table_cells(self):
        evidence, images = hybrid.build_evidence(document(), question(), 8)
        self.assertEqual([ref for ref, _ in images], ["#/pages/1", "#/pictures/0"])
        self.assertIn("#/texts/1", evidence["sources"])
        self.assertEqual(evidence["sources"]["#/tables/0"]["data"]["table_cells"][0]["text"], "Table option")
        self.assertNotIn("base64", json.dumps(evidence))

    def test_image_cap_and_nonembedded_image_warnings(self):
        evidence, images = hybrid.build_evidence(document(), question(), 1)
        self.assertEqual(len(images), 1)
        self.assertTrue(evidence["evidence_warnings"])
        doc = document()
        doc["pages"]["1"]["image"]["uri"] = "https://example.com/image.png"
        evidence, images = hybrid.build_evidence(doc, question(), 8)
        self.assertEqual([ref for ref, _ in images], ["#/pictures/0"])
        self.assertTrue(evidence["evidence_warnings"])

    def test_rejects_wrong_identity_answer_duplicate_labels_and_unknown_images(self):
        for update in [
            {"question_number": "2"}, {"correct_answer": "C"},
            {"image_refs": ["#/pictures/999"]},
            {"options": [{"label": "A", "text": "x", "image_refs": []}] * 2},
        ]:
            value = extraction()
            value.update(update)
            with self.subTest(update=update), self.assertRaises(ValueError):
                hybrid.validate_response(value, question(), set())

    def test_accepts_graphical_option_and_unknown_answer(self):
        value = extraction()
        value["correct_answer"] = None
        value["options"][0].update(text="", image_refs=["#/pictures/0"])
        self.assertEqual(hybrid.validate_response(value, question(), {"#/pictures/0"}), value)

    def test_api_request_sends_images_and_validates_response(self):
        formula_result = extraction()
        formula_result["question_text"] = r"Find $10^{5}$ for $R_{1}$ and $\sqrt{x}$."
        formula_result["options"][0]["text"] = r"$\frac{a}{b}$"
        payload = {"id": "response-id", "model": "test/vision", "usage": {},
                   "choices": [{"finish_reason": "stop", "message": {
                       "content": json.dumps(formula_result)}}]}
        evidence, images = hybrid.build_evidence(document(), question(), 8)
        with patch.object(review_api, "urlopen", return_value=io.BytesIO(json.dumps(payload).encode())) as send:
            result = hybrid.call_openrouter(evidence, images, question(), api_key="fake",
                                             model="test/vision", retries=0, timeout=1)
        request = send.call_args.args[0]
        body = json.loads(request.data)
        self.assertTrue(body["response_format"]["json_schema"]["strict"])
        content = body["messages"][1]["content"]
        self.assertEqual(sum(item["type"] == "image_url" for item in content), 2)
        self.assertEqual(result["extraction"]["correct_answer"], "B")
        self.assertEqual(result["extraction"]["question_text"], formula_result["question_text"])
        self.assertEqual(result["extraction"]["options"][0]["text"], r"$\frac{a}{b}$")

    def test_truncated_model_response_is_rejected(self):
        payload = {"choices": [{"finish_reason": "length", "message": {
            "content": json.dumps(extraction())}}]}
        with patch.object(review_api, "urlopen", return_value=io.BytesIO(json.dumps(payload).encode())):
            with self.assertRaisesRegex(ValueError, "did not finish normally"):
                hybrid.call_openrouter({}, [], question(), api_key="fake", model="test/vision",
                                       retries=0, timeout=1)

    def test_selective_processing_promotes_candidates_preserves_failures_and_resumes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw = root / "raw.json"
            mcqs = root / "mcqs.json"
            raw.write_text(json.dumps({"document": document()}))
            unflagged = question("0", False)
            flagged = question("1")
            candidate = question("2")
            mcqs.write_text(json.dumps({"questions": [unflagged, flagged],
                                       "review_candidates": [candidate]}))
            args = argparse.Namespace(raw_json=raw, mcq_json=mcqs,
                output=root / "out.json", model="test/vision", max_images=8,
                limit=None, retries=0, timeout=1, resume=False, dry_run=False, review_all=False)
            def response(*_, **kwargs):
                return {"extraction": extraction(), "usage": {}, "response_id": "test", "model": "test/vision"}
            with patch.dict("os.environ", {"OPENROUTER_API_KEY": "fake-test-key"}), \
                 patch.object(hybrid, "call_openrouter", side_effect=[response(), RuntimeError("offline failure")]) as call:
                hybrid.run(args)
            self.assertEqual(call.call_count, 2)
            first = json.loads(args.output.read_text())
            self.assertEqual(first["questions"][0], unflagged)
            self.assertFalse(first["questions"][1]["needs_review"])
            self.assertEqual(first["review_candidates"][0]["question_text"], "Draft")
            self.assertIn("llm_error", first["review_candidates"][0])

            args.resume = True
            recovered = response()
            recovered["extraction"]["question_number"] = "2"
            with patch.dict("os.environ", {"OPENROUTER_API_KEY": "fake-test-key"}), \
                 patch.object(hybrid, "call_openrouter", return_value=recovered) as call:
                hybrid.run(args)
            self.assertEqual(call.call_count, 1)
            final = json.loads(args.output.read_text())
            self.assertEqual(final["question_count"], 3)
            self.assertEqual(final["review_candidates"], [])
            self.assertEqual(final["questions"][2]["source_refs"], candidate["source_refs"])


if __name__ == "__main__":
    unittest.main()
