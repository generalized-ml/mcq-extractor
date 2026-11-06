"""Timing persistence and CSV plumbing; no scoring, models, or evaluations run."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation.evaluate import cost_per_pdf_report, load_combined, save_results
from evaluation.plot_experiments import read_experiments
from pdf_to_jsons.runtime import RuntimeRecorder, combine_runtimes, recorded_runtime


class RuntimeReportingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_snapshots_do_not_double_count_and_resume_excludes_idle_gap(self):
        with patch('pdf_to_jsons.runtime.time.perf_counter', side_effect=[10, 15, 19, 100, 104]):
            first = RuntimeRecorder(self.root, 'pdf_to_json_my')
            first.save()
            first.save(finished=True)
            resumed = RuntimeRecorder(self.root, 'pdf_to_json_my')
            resumed.save(finished=True)
        report = recorded_runtime(self.root / 'final_mcqs.json', 'pdf_to_json_my')
        self.assertEqual(report['method_wall_seconds'], 13)
        self.assertEqual(report['method_unfinished_time_pdfs'], 0)
        self.assertEqual(len(json.loads((self.root / 'runtime.json').read_text())['sessions']), 2)

    def test_original_nested_output_finds_pdf_sidecar(self):
        recorder = RuntimeRecorder(self.root, 'pdf_to_json', started=0)
        with patch('pdf_to_jsons.runtime.time.perf_counter', return_value=7):
            recorder.save(finished=True)
        report = recorded_runtime(self.root / 'mcq_with_llm' / '2021_mcqs_with_llm.json', 'pdf_to_json')
        self.assertEqual(report['method_wall_seconds'], 7)

    def test_missing_time_stays_unknown_and_method_labels_are_checked(self):
        report = recorded_runtime(self.root / 'final_mcqs.json', 'pdf_to_json_my')
        self.assertIsNone(report['method_wall_seconds'])
        self.assertEqual(report['method_missing_time_pdfs'], 1)
        RuntimeRecorder(self.root, 'pdf_to_json_paddle').save(finished=True)
        self.assertIsNone(recorded_runtime(self.root / 'final_mcqs.json', 'pdf_to_json_my')['method_wall_seconds'])

    def test_combined_reports_expose_missing_and_incomplete_timings(self):
        recorder = RuntimeRecorder(self.root, 'pdf_to_json_my', started=0)
        with patch('pdf_to_jsons.runtime.time.perf_counter', return_value=12):
            recorder.save()
        known = recorded_runtime(self.root / 'final_mcqs.json', 'pdf_to_json_my')
        missing = recorded_runtime(self.root / 'missing' / 'nested' / 'final_mcqs.json', 'pdf_to_json_my')
        combined = combine_runtimes([known, missing])
        self.assertEqual(combined['method_wall_seconds'], 12)
        self.assertEqual(combined['method_timed_pdfs'], 1)
        self.assertEqual(combined['method_missing_time_pdfs'], 1)
        self.assertEqual(combined['method_unfinished_time_pdfs'], 1)

    def test_invalid_duration_is_rejected(self):
        for value in (-1, True, float('nan'), float('inf'), '12'):
            with self.subTest(value=value):
                (self.root / 'runtime.json').write_text(json.dumps({
                    'method': 'pdf_to_json_my', 'sessions': [{'seconds': value}]}))
                with self.assertRaisesRegex(ValueError, 'Invalid runtime duration'):
                    recorded_runtime(self.root / 'final_mcqs.json', 'pdf_to_json_my')

    def test_first_timed_resume_exposes_unrecorded_earlier_work(self):
        RuntimeRecorder(self.root, 'pdf_to_json_my', history_complete=False).save(finished=True)
        RuntimeRecorder(self.root, 'pdf_to_json_my').save(finished=True)
        report = recorded_runtime(self.root / 'final_mcqs.json', 'pdf_to_json_my')
        self.assertEqual(report['method_incomplete_history_pdfs'], 1)

    def test_csv_upgrade_preserves_existing_scores_and_leaves_old_time_blank(self):
        path = self.root / 'experiment.csv'
        path.write_text('experiment_number,experiment_name,keyword_score\n1,pdf_to_json,0.7\n')
        number, _ = save_results(self.root, [], {'keyword_score': .9, 'method_wall_seconds': 42,
                                               'evaluation_wall_seconds': 2},
                                 {'experiment_name': 'pdf_to_json_my'}, None)
        self.assertEqual(number, 2)
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[0]['keyword_score'], '0.7')
        self.assertEqual(rows[0]['method_wall_seconds'], '')
        self.assertEqual(float(rows[1]['method_wall_seconds']), 42)
        self.assertEqual(float(rows[1]['evaluation_wall_seconds']), 2)

    def test_unrelated_csv_schema_is_preserved(self):
        path = self.root / 'experiment.csv'
        original = 'experiment_number,unexpected\n1,value\n'
        path.write_text(original)
        with self.assertRaisesRegex(ValueError, 'different schema'):
            save_results(self.root, [], {'keyword_score': .9}, {'experiment_name': 'pdf_to_json_my'}, None)
        self.assertEqual(path.read_text(), original)

    def test_cost_per_pdf_averages_available_costs_and_exposes_unknowns(self):
        report = cost_per_pdf_report([{'llm_cost_usd': .1}, {'llm_cost_usd': .3},
                                      {'llm_cost_usd': None}], 4)
        self.assertAlmostEqual(report['llm_cost_per_pdf_usd'], .2)
        self.assertEqual(report['llm_cost_pdf_count'], 2)
        self.assertEqual(report['llm_missing_cost_pdfs'], 2)
        self.assertEqual(cost_per_pdf_report([{'llm_cost_usd': 0}], 1)['llm_cost_per_pdf_usd'], 0)
        self.assertIsNone(cost_per_pdf_report([{'llm_cost_usd': None}], 1)['llm_cost_per_pdf_usd'])

    def test_combined_cost_uses_pdf_count_not_question_count(self):
        golden_root = self.root / 'golden'
        inputs = self.root / 'inputs'
        question = {'question_number': '1', 'question_text': 'Fixture',
                    'options': ['a', 'b', 'c', 'd'], 'correct_answer': '1'}
        for year in (2021, 2022, 2023):
            folder = golden_root / f'neet_{year}'
            folder.mkdir(parents=True)
            (folder / f'{year}_mcqs.json').write_text(json.dumps([question]))
        for year, cost in ((2021, .1), (2022, .3)):
            folder = inputs / str(year)
            folder.mkdir(parents=True)
            (folder / 'final_mcqs.json').write_text(json.dumps({
                'questions': [question], 'llm_calls': [{'usage': {'cost': cost}}]}))
        _, _, summary, _, _, _ = load_combined(inputs, 'pdf_to_json_my', golden_root, [2021, 2022, 2023])
        self.assertAlmostEqual(summary['llm_cost_per_pdf_usd'], .2)
        self.assertEqual(summary['llm_cost_pdf_count'], 2)
        self.assertEqual(summary['llm_missing_cost_pdfs'], 1)

    def test_cost_schema_upgrade_preserves_legacy_total_without_relabeling(self):
        path = self.root / 'experiment.csv'
        path.write_text('experiment_number,experiment_name,keyword_score,llm_cost_usd\n1,pdf_to_json,0.7,5\n')
        save_results(self.root, [], {'keyword_score': .9, 'llm_cost_usd': .4,
                                     'llm_cost_per_pdf_usd': .2, 'llm_cost_pdf_count': 2},
                     {'experiment_name': 'pdf_to_json_my'}, None)
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertNotIn('llm_cost_usd', rows[1])
        self.assertEqual(rows[0]['legacy_llm_cost_usd'], '5')
        self.assertEqual(rows[0]['llm_cost_per_pdf_usd'], '')
        self.assertEqual(float(rows[1]['llm_cost_per_pdf_usd']), .2)
        self.assertEqual(rows[1]['legacy_llm_cost_usd'], '')

    def test_plot_reads_cost_per_pdf_and_flags_partial_pdf_coverage(self):
        path = self.root / 'experiment.csv'
        path.write_text('experiment_number,experiment_name,keyword_score,semantic_score,'
                        'llm_cost_per_pdf_usd,llm_missing_cost_pdfs\n1,pdf_to_json_my,0.8,0.9,0.2,1\n')
        row = read_experiments(path)[0]
        self.assertEqual(row['cost'], .2)
        self.assertTrue(row['partial_cost'])
