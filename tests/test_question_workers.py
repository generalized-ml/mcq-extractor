import unittest
from types import SimpleNamespace
from unittest.mock import patch

from mcq_extractor.question_workers import execution_settings


class WorkerAllocationTests(unittest.TestCase):
    def settings(self, cores, questions, **kwargs):
        args = SimpleNamespace(physical_cores=None, workers=None, device='cpu')
        args.__dict__.update(kwargs)
        with patch('psutil.cpu_count', return_value=cores) as detect:
            settings = execution_settings(args, questions)
        if args.physical_cores is None:
            detect.assert_called_once_with(logical=False)
        return settings

    def test_uses_two_physical_cores_per_worker(self):
        result = self.settings(16, 200)
        self.assertEqual((result['workers'], result['threads_per_worker']), (8, 2))
        self.assertEqual(result['questions_per_worker_ceiling'], 25)

    def test_caps_workers_to_pending_questions(self):
        self.assertEqual(self.settings(16, 3)['workers'], 3)
        self.assertEqual(self.settings(16, 0)['workers'], 0)

    def test_small_odd_and_unknown_core_counts(self):
        self.assertEqual(self.settings(1, 10)['threads_per_worker'], 1)
        self.assertEqual(self.settings(3, 10)['workers'], 1)
        with self.assertRaisesRegex(ValueError, 'physical CPU cores'):
            self.settings(None, 10)
        self.assertEqual(self.settings(None, 10, physical_cores=8)['workers'], 4)

    def test_gpu_defaults_to_one_worker_and_rejects_multiple(self):
        self.assertEqual(self.settings(16, 20, device='mps')['workers'], 1)
        with self.assertRaisesRegex(ValueError, 'requires --device cpu'):
            self.settings(16, 20, device='mps', workers=2)
