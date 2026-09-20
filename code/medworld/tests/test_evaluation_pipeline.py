import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

from medworld import evaluate_run, launch_distributed
from medworld.run_experiment import evaluation_jobs
from medworld.config import load_config
from medworld.datasets.protocol import _sha256


class EvaluationPipelineTests(unittest.TestCase):
    def test_full_test_plan_has_only_three_tasks_and_human_segmentation(self):
        jobs = evaluation_jobs(Path('/run'))
        tests = [j for j in jobs if j['module'].endswith('.evaluate')]
        self.assertEqual(len(tests), 4)
        for job in tests:
            self.assertIn('/run/final.pt', job['args'])
            self.assertNotIn('--limit', job['args'])

    def prepare(self, root):
        (root / 'status.json').write_text(json.dumps({'complete': True, 'stopped': False}))
        (root / 'final.pt').touch()
        (root / 'source_manifest.json').write_text('{}')
        (root / 'config.json').write_text(json.dumps(load_config()))
        (root / 'data_protocol.json').write_text('{}')

    def test_failed_evaluation_preserves_training_and_records_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.prepare(root)
            with patch.object(evaluate_run, 'registry') as registry, patch.object(evaluate_run, 'schedule', side_effect=RuntimeError('test failure')):
                with self.assertRaisesRegex(RuntimeError, 'test failure'):
                    evaluate_run.evaluate_run(root, ['1'])
            state = json.loads((root / 'pipeline_status.json').read_text())
            self.assertTrue(state['training_complete'])
            self.assertFalse(state['evaluation_complete'])
            self.assertEqual(state['phase'], 'failed')
            self.assertTrue((root / 'final.pt').exists())
            self.assertEqual(registry.call_args.args[1], 'finished')

    def test_testing_config_selects_or_disables_jobs(self):
        cfg = load_config(overrides={'testing': {'tasks': ['vqa']}})
        self.assertEqual([j['id'] for j in evaluation_jobs(Path('/run'), cfg)], ['vqa'])
        cfg = load_config(overrides={'testing': {'tasks': ['segmentation'], 'human_segmentation': False}})
        self.assertEqual([j['id'] for j in evaluation_jobs(Path('/run'), cfg)], ['segmentation'])
        cfg = load_config(overrides={'testing': {'enabled': False, 'tasks': []}})
        self.assertEqual(evaluation_jobs(Path('/run'), cfg), [])
        for testing in ({'enabled': 'false'}, {'tasks': ['sr']}, {'tasks': ['vqa', 'vqa']}, {'tasks': []}, {'unknown': True}):
            with self.subTest(testing=testing), self.assertRaises(ValueError):
                load_config(overrides={'testing': testing})

    def test_disabled_testing_never_starts_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.prepare(root)
            (root / 'config.json').write_text(json.dumps(load_config(overrides={'testing': {'enabled': False}})))
            with patch.object(evaluate_run, 'schedule') as schedule:
                evaluate_run.evaluate_run(root, ['1'])
                schedule.assert_not_called()
            self.assertTrue(json.loads((root / 'pipeline_status.json').read_text())['evaluation_skipped'])

    def test_reuse_checks_checkpoint_and_prediction_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = load_config(overrides={'testing': {'tasks': ['vqa']}})
            job = evaluation_jobs(root, cfg)[0]
            directory = root / 'evaluation/vqa'
            directory.mkdir(parents=True)
            predictions = directory / 'vqa.jsonl'
            predictions.write_text('{"id":"x"}\n')
            summary = {'checkpoint_sha256': 'checkpoint', 'data_fingerprint': 'data', 'limit': None,
                       'split': 'test', 'tasks': {'vqa': {'n': 1}}, 'predictions_sha256': {'vqa': _sha256(predictions)}}
            (directory / 'summary.json').write_text(json.dumps(summary))
            self.assertTrue(evaluate_run.completed_job(root, job, 'checkpoint', 'data'))
            with self.assertRaisesRegex(ValueError, 'checkpoint/protocol'):
                evaluate_run.completed_job(root, job, 'different', 'data')
            predictions.write_text('{"id":"tampered"}\n')
            with self.assertRaisesRegex(ValueError, 'contents changed'):
                evaluate_run.completed_job(root, job, 'checkpoint', 'data')

    def test_incomplete_training_is_never_tested(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.prepare(root)
            (root / 'status.json').write_text('{"complete": false}')
            with patch.object(evaluate_run, 'schedule') as schedule:
                with self.assertRaises(ValueError):
                    evaluate_run.evaluate_run(root, ['1'])
                schedule.assert_not_called()

    def test_launcher_evaluates_by_default_after_releasing_gpu_locks(self):
        for skip, complete, returncode in ((False, True, 0), (True, True, 0), (False, False, 0), (False, True, 1)):
            with self.subTest(skip=skip, complete=complete, returncode=returncode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / 'run'
                locks = [Mock(), Mock()]
                child = Mock(pid=12345)
                child.poll.return_value = returncode
                child.wait.return_value = returncode
                def snapshot(out):
                    (out / 'status.json').write_text(json.dumps({'complete': complete}))
                def check_locks(*_):
                    for lock in locks:
                        lock.close.assert_called_once()
                argv = ['launch', '--gpus', '1,2', '--out', str(root)] + (['--skip-evaluation'] if skip else [])
                with patch('sys.argv', argv), patch.object(launch_distributed, 'reserve', return_value=(locks, [{'uuid': 'a'}, {'uuid': 'b'}])), patch.object(launch_distributed, 'snapshot', side_effect=snapshot), patch.object(launch_distributed.subprocess, 'Popen', return_value=child), patch.object(launch_distributed.signal, 'signal'), patch.object(evaluate_run, 'evaluate_run', side_effect=check_locks) as evaluate:
                    with self.assertRaises(SystemExit) as result:
                        launch_distributed.main()
                    self.assertEqual(result.exception.code, returncode)
                self.assertEqual(evaluate.call_count, int(not skip and complete and returncode == 0))


if __name__ == '__main__':
    unittest.main()
