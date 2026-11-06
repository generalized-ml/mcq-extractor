"""Offline checks for scoring, alignment, normalization, and experiment CSVs."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import contextlib
import io

from evaluation.evaluate import (
    WEIGHTS, cosine_similarity, evaluate, keyword_match,
    load_questions, normalize_question, save_results, llm_cost_report, main,
    find_golden_json,
    load_combined,
)


def question(number="1", **changes):
    value = {"question_number": number, "question_text": "Find the resistance.",
             "options": ["2 ohm", "4 ohm", "6 ohm", "8 ohm"], "correct_answer": "2"}
    value.update(changes)
    return normalize_question(value, 1)


def fake_encoder(texts):
    # Deterministic vectors to test metric plumbing without downloading a model.
    return [[1.0, 0.0] for _ in texts]


class EvaluationTests(unittest.TestCase):
    def test_combined_years_keep_question_ids_separate_and_score_missing_year(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            golden_root, predictions_root = root / 'golden', root / 'predictions'
            for year in (2021, 2022):
                folder = golden_root / f'neet_{year}'
                folder.mkdir(parents=True)
                (folder / f'{year}_mcqs.json').write_text(json.dumps([{
                    'question_number': '1', 'question_text': f'Year {year}',
                    'options': ['a', 'b', 'c', 'd'], 'correct_answer': '1'}]))
            output = predictions_root / '2021'
            output.mkdir(parents=True)
            (output / 'final_mcqs.json').write_text(json.dumps({
                'questions': [{'question_number': '1', 'question_text': 'Year 2021',
                               'options': ['a', 'b', 'c', 'd'], 'correct_answer': '1'}],
                'llm_calls': [{'response_id': 'call-1', 'usage': {'cost': .01}}]}))
            actual, golden, cost, _, _, coverage = load_combined(
                predictions_root, 'pdf_to_json_my', golden_root, [2021, 2022])
            self.assertEqual([item['identity'] for item in golden], ['2021:1', '2022:1'])
            self.assertEqual([item['identity'] for item in actual], ['2021:1'])
            self.assertEqual(coverage, [(2021, 1, 1, True), (2022, 0, 1, False)])
            self.assertAlmostEqual(cost['llm_cost_usd'], .01)
            _, summary = evaluate(actual, golden, fake_encoder)
            self.assertEqual(summary['missing_predictions'], 1)
            self.assertAlmostEqual(summary['keyword_score'], .5)

    def test_combined_original_discards_numbered_cover_page_instruction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            golden = root / 'golden' / 'neet_2024'
            golden.mkdir(parents=True)
            complete = {'question_number': '1', 'question_text': 'Actual question',
                        'options': ['a', 'b', 'c', 'd'], 'correct_answer': '2'}
            (golden / '2024_mcqs.json').write_text(json.dumps([complete]))
            source = root / 'outputs' / '2024' / 'mcq_with_llm'
            source.mkdir(parents=True)
            (source / '2024_mcqs_with_llm.json').write_text(json.dumps({'questions': [
                {'question_number': '1', 'question_text': 'Instructions', 'options': ['a', 'b']},
                complete]}))
            actual, _, _, _, _, coverage = load_combined(
                root / 'outputs', 'pdf_to_json', root / 'golden', [2024])
            self.assertEqual(len(actual), 1)
            self.assertEqual(actual[0]['fields']['question'], 'Actual question')
            self.assertEqual(coverage[0][:3], (2024, 1, 1))

    def test_infers_renamed_neet_golden_folder(self):
        expected = Path('dataset/golden/neet_2021/2021_mcqs.json').resolve()
        self.assertEqual(find_golden_json(Path('outputs/json_crops_paddle/2021/final_mcqs.json').resolve()),
                         expected)

    def test_cost_uses_checkpoint_once_and_includes_incomplete_calls(self):
        record = {"status": "success", "response_id": "call-1", "model": "vision",
                  "usage": {"cost": 0.0004313, "prompt_tokens": 100,
                            "completion_tokens": 20, "total_tokens": 120}}
        payload = {"questions": [{"llm": record}], "hybrid": {"results": {
            "questions/0": record,
            "review_candidates/0": {**record, "response_id": "call-2",
                                    "usage": {"cost": 0.0002, "total_tokens": 50}},
            "questions/1": {"status": "error", "error": "timeout"},
        }}}
        calls, summary = llm_cost_report(payload)
        self.assertEqual(len(calls), 3)
        self.assertAlmostEqual(summary["llm_cost_usd"], 0.0006313)
        self.assertEqual(summary["llm_priced_calls"], 2)
        self.assertEqual(summary["llm_missing_cost_calls"], 1)
        self.assertEqual(summary["llm_total_tokens"], 170)

    def test_cost_fallback_deduplicates_response_ids(self):
        record = {"response_id": "same", "usage": {"cost": "0.001"}}
        calls, summary = llm_cost_report({"questions": [{"llm": record}],
                                         "review_candidates": [{"llm": record}]})
        self.assertEqual(len(calls), 1)
        self.assertEqual(summary["llm_cost_usd"], 0.001)
        with self.assertRaisesRegex(ValueError, "Conflicting usage"):
            llm_cost_report([{"llm": record}, {"llm": {**record, "usage": {"cost": 1}}}])

    def test_unknown_and_invalid_costs_are_not_reported_as_free(self):
        for value in (None, True, -1, "NaN", "Infinity", "invalid"):
            with self.subTest(value=value):
                _, summary = llm_cost_report([{"llm": {"usage": {"cost": value}}}])
                self.assertIsNone(summary["llm_cost_usd"])
                self.assertEqual(summary["llm_missing_cost_calls"], 1)
        _, summary = llm_cost_report([{"llm": {"usage": {"cost": 0}}}])
        self.assertEqual(summary["llm_cost_usd"], 0)
        self.assertEqual(summary["llm_priced_calls"], 1)
        self.assertIsNone(llm_cost_report([{}])[1]["llm_cost_usd"])

    def test_cli_saves_only_one_experiment_csv_with_costs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = {"question_number": "1", "question_text": "Example",
                      "options": ["a", "b", "c", "d"]}
            prediction, golden = root / "prediction.json", root / "golden.json"
            prediction.write_text(json.dumps({"questions": [record], "hybrid": {"results": {
                "questions/0": {"status": "success", "model": "test/vision",
                                "response_id": "call-1", "usage": {"cost": 0.0004313,
                                "prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}}
            }}}))
            golden.write_text(json.dumps([record]))
            output = root / "results"
            with patch("sys.argv", ["evaluate", "--mcq-json", str(prediction),
                                    "--golden-json", str(golden), "--output-dir", str(output)]), \
                 patch("evaluation.evaluate.SentenceEncoder", return_value=fake_encoder), \
                 contextlib.redirect_stdout(io.StringIO()):
                main()
            with (output / "experiment.csv").open() as stream:
                summary = next(csv.DictReader(stream))
            self.assertAlmostEqual(float(summary["llm_cost_per_pdf_usd"]), 0.0004313)
            self.assertEqual(summary['llm_cost_pdf_count'], '1')
            self.assertNotIn('llm_cost_usd', summary)
            self.assertEqual(summary["llm_total_tokens"], "120")
            self.assertEqual(summary["llm_priced_calls"], "1")
            self.assertEqual([p.name for p in output.iterdir()], ["experiment.csv"])

    def test_keyword_stops_at_first_different_word(self):
        result = keyword_match("one WRONG three four", "one two three four")
        self.assertEqual(result["score"], 0.25)
        self.assertEqual(result["first_mismatch_index"], 2)
        self.assertEqual(result["expected_word"], "two")
        self.assertEqual(result["actual_word"], "WRONG")

    def test_keyword_whitespace_case_punctuation_and_suffix(self):
        self.assertEqual(keyword_match("a\n b", "a b")["score"], 1.0)
        self.assertEqual(keyword_match("A b", "a b")["score"], 0.0)
        self.assertEqual(keyword_match("a b!", "a b")["score"], 0.5)
        self.assertEqual(keyword_match("a b extra", "a b")["score"], 1.0)
        self.assertEqual(keyword_match("", "a b")["score"], 0.0)
        self.assertEqual(keyword_match("", "")["score"], 0.0)

    def test_correct_answer_is_solution_even_if_explanation_exists(self):
        q = question(correct_answer=3, solution="Do not compare this explanation")
        self.assertEqual(q["fields"]["solution"], "3")
        self.assertEqual(question(correct_answer=None)["fields"]["solution"], "")

    def test_normalizes_option_labels_and_aligns_questions_by_id(self):
        value = question(options=[{"label": "B", "text": "second"},
                                  {"label": "A", "text": "first"}])
        self.assertEqual(value["fields"]["option_1"], "first")
        self.assertEqual(value["fields"]["option_2"], "second")
        rows, _ = evaluate([question("2"), question("1")], [question("1"), question("2")], fake_encoder)
        self.assertTrue(rows[0]["identifiers_match"])
        self.assertEqual(rows[0]["prediction_question_id"], "1")
        self.assertEqual(rows[1]["prediction_question_id"], "2")

    def test_missing_middle_id_scores_zero_without_shifting_following_questions(self):
        rows, summary = evaluate([question("3"), question("1")],
                                 [question("1"), question("2"), question("3")], fake_encoder)
        self.assertTrue(rows[1]["prediction_missing"])
        self.assertEqual(rows[1]["prediction_question_id"], "")
        for field in WEIGHTS:
            self.assertEqual(rows[1][f"{field}_keyword_score"], 0.0)
            self.assertEqual(rows[1][f"{field}_semantic_score"], 0.0)
        self.assertEqual(rows[1]["keyword_score"], 0.0)
        self.assertEqual(rows[1]["semantic_score"], 0.0)
        self.assertEqual(rows[2]["prediction_question_id"], "3")
        self.assertAlmostEqual(rows[2]["keyword_score"], 1.0)
        self.assertAlmostEqual(summary["keyword_score"], 2 / 3)

    def test_counts_unmatched_ids_even_when_array_lengths_match(self):
        rows, summary = evaluate([question("1"), question("4")],
                                 [question("1"), question("2")], fake_encoder)
        self.assertEqual(summary["missing_predictions"], 1)
        self.assertEqual(summary["extra_predictions"], 1)
        self.assertEqual(rows[1]["semantic_score"], 0.0)

    def test_missing_prediction_id_cannot_match_by_position(self):
        rows, summary = evaluate([question(None)], [question("1")], fake_encoder)
        self.assertEqual(rows[0]["keyword_score"], 0.0)
        self.assertEqual(rows[0]["semantic_score"], 0.0)
        self.assertEqual(summary["extra_predictions"], 1)

    def test_duplicate_ids_and_missing_golden_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate prediction"):
            evaluate([question("1"), question("1")], [question("1")], fake_encoder)
        with self.assertRaisesRegex(ValueError, "Duplicate golden"):
            evaluate([question("1")], [question("1"), question("1")], fake_encoder)
        with self.assertRaisesRegex(ValueError, "Every golden"):
            evaluate([question("1")], [question(None)], fake_encoder)

    def test_duplicate_predictions_are_flagged_and_scored_empty(self):
        value = question(options=[{"label": "1", "text": "a"}, {"label": "1", "text": "b"}])
        self.assertEqual(value["fields"]["option_1"], "")
        self.assertTrue(value["schema_warnings"])
        with self.assertRaises(ValueError):
            normalize_question({"options": [{"label": "1"}, {"label": "1"}]}, 1, strict=True)

    def test_weights_and_missing_questions(self):
        rows, summary = evaluate([question()], [question(), question("2")], fake_encoder)
        self.assertAlmostEqual(sum(WEIGHTS.values()), 1.0)
        self.assertAlmostEqual(rows[0]["keyword_score"], 1.0)
        self.assertAlmostEqual(rows[0]["semantic_score"], 1.0)
        self.assertEqual(rows[1]["keyword_score"], 0.0)
        self.assertEqual(rows[1]["semantic_score"], 0.0)
        self.assertAlmostEqual(summary["keyword_score"], 0.5)
        self.assertEqual(summary["missing_predictions"], 1)

    def test_question_options_and_answer_weights(self):
        rows, _ = evaluate([question(options=["wrong", "4 ohm", "6 ohm", "8 ohm"],
                                     correct_answer="wrong")], [question()], fake_encoder)
        self.assertAlmostEqual(WEIGHTS["question"], .3)
        self.assertTrue(all(WEIGHTS[f"option_{slot}"] == .15 for slot in range(1, 5)))
        self.assertAlmostEqual(WEIGHTS["solution"], .1)
        self.assertAlmostEqual(rows[0]["keyword_score"], .75)
        self.assertAlmostEqual(rows[0]["aggregate_score"],
                               (rows[0]["keyword_score"] + rows[0]["semantic_score"]) / 2)

    def test_semantic_pairs_use_corresponding_fields_and_weights(self):
        def encode(texts):
            self.assertEqual(len(texts), len(set(texts)))
            return [[0, 1] if text == "different option" else [1, 0] for text in texts]
        rows, summary = evaluate(
            [question(options=["different option", "4 ohm", "6 ohm", "8 ohm"])],
            [question()], encode,
        )
        self.assertEqual(rows[0]["option_1_semantic_score"], 0.0)
        self.assertEqual(rows[0]["question_semantic_score"], 1.0)
        self.assertAlmostEqual(summary["semantic_score"], 0.85)
        self.assertAlmostEqual(summary["aggregate_score"],
                               (summary["keyword_score"] + summary["semantic_score"]) / 2)

    def test_empty_fields_get_zero_and_extra_predictions_are_reported(self):
        rows, summary = evaluate([question(correct_answer=None), question("2")],
                                 [question(correct_answer=None)], fake_encoder)
        self.assertAlmostEqual(rows[0]["keyword_score"], 0.9)
        self.assertAlmostEqual(rows[0]["semantic_score"], 0.9)
        self.assertEqual(summary["extra_predictions"], 1)
        self.assertEqual(summary["solution_golden_empty_count"], 1)

    def test_empty_candidate_dataset_and_empty_golden(self):
        _, summary = evaluate([], [question()], fake_encoder)
        self.assertEqual(summary["keyword_score"], 0.0)
        self.assertEqual(summary["semantic_score"], 0.0)
        with self.assertRaises(ValueError):
            evaluate([], [], fake_encoder)

    def test_cosine_and_clipping(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [1, 1]), 2 ** -0.5)
        self.assertEqual(cosine_similarity([1, 0], [-1, 0]), 0.0)
        self.assertEqual(cosine_similarity([0, 0], [1, 0]), 0.0)

    def test_shared_csv_distinguishes_pipelines(self):
        rows, summary = evaluate([question()], [question()], fake_encoder)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('pdf_to_json', 'pdf_to_json_paddle', 'pdf_to_json'):
                save_results(root, rows, summary, {'experiment_name': name}, None)
            with (root / 'experiment.csv').open() as stream:
                records = list(csv.DictReader(stream))
            self.assertEqual([r['experiment_number'] for r in records], ['1', '2', '3'])
            self.assertEqual([r['experiment_name'] for r in records],
                             ['pdf_to_json', 'pdf_to_json_paddle', 'pdf_to_json'])
            self.assertEqual(len(list(root.iterdir())), 1)

    def test_csv_experiment_numbers_and_overwrite_protection(self):
        rows, summary = evaluate([question()], [question()], fake_encoder)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            first, path = save_results(output, rows, summary, {"model": "test"}, None)
            second, _ = save_results(output, rows, summary, {"model": "test"}, None)
            self.assertEqual((first, second), (1, 2))
            with path.open() as stream:
                result = list(csv.DictReader(stream))
            self.assertEqual(len(result), 2)
            self.assertEqual([p.name for p in output.iterdir()], ["experiment.csv"])
            self.assertEqual(result[0]["keyword_score"], "1.00000000")
            self.assertEqual(result[0]["experiment_number"], "1")
            with self.assertRaises(ValueError):
                save_results(output, rows, summary, {"model": "test"}, 1)

    def test_accepts_array_and_pipeline_json_and_simple_schema(self):
        record = {"question_id": "PHY_001", "text": "Find resistance", "options": ["a", "b", "c", "d"],
                  "correct_answer": 2}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "questions.json"
            path.write_text(json.dumps([record]))
            expected = load_questions(path)
            path.write_text(json.dumps({"questions": [record]}))
            self.assertEqual(load_questions(path), expected)


if __name__ == "__main__":
    unittest.main()
