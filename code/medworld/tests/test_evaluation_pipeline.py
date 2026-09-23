import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

from medworld import evaluate_run, launch_distributed
from medworld.run_experiment import evaluation_jobs
from medworld.config import load_config
from medworld.datasets.protocol import _sha256
from medworld.evaluation.protocol import FUTURE_TASKS, metric_protocol
from medworld.evaluation.selection import reference_manifest, select_vqa
from medworld.evaluation.future_metrics import reference_fingerprint, score_future_task
from medworld.evaluation.future_common import scoring_protocol
from medworld.datasets.vqa import VOCABULARY


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
            records = [{'id': 'x', 'patient': 'p', 'question': 'Question?', 'answer': ['yes'],
                        'semantic_type': 'verify', 'prediction': ['yes']}]
            predictions.write_text(json.dumps(records[0]) + '\n')
            _, selection = select_vqa(records)
            summary = {'checkpoint_sha256': 'checkpoint', 'data_fingerprint': 'data', 'limit': None,
                       'metric_protocol': metric_protocol(), 'references': {'vqa': reference_manifest('vqa', records)},
                       'vqa_selection': selection, 'split': 'test', 'tasks': {'vqa': {'n': 1}},
                       'predictions_sha256': {'vqa': _sha256(predictions)}}
            (directory / 'summary.json').write_text(json.dumps(summary))
            self.assertTrue(evaluate_run.completed_job(root, job, 'checkpoint', 'data'))
            with self.assertRaisesRegex(ValueError, 'checkpoint/protocol'):
                evaluate_run.completed_job(root, job, 'different', 'data')
            predictions.write_text('{"id":"tampered"}\n')
            with self.assertRaisesRegex(ValueError, 'contents changed'):
                evaluate_run.completed_job(root, job, 'checkpoint', 'data')

    def test_future_enabled_plan_tests_every_table_column(self):
        cfg = load_config(overrides={'future_enabled': True})
        jobs = evaluation_jobs(Path('/run'), cfg)
        self.assertEqual({job['id'] for job in jobs},
                         {'classification', 'segmentation', 'segmentation_human', 'vqa', *FUTURE_TASKS})
        self.assertTrue(all('--limit' not in job['args'] for job in jobs))

    def test_future_reuse_requires_current_scorer_files_and_dataset_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = load_config(overrides={'future_enabled': True})
            (root / 'config.json').write_text(json.dumps(cfg))
            data_protocol = {'future': {'cohort': 'fixed'}}
            (root / 'data_protocol.json').write_text(json.dumps(data_protocol))
            job = next(job for job in evaluation_jobs(root, cfg) if job['id'] == 'future_vqa')
            directory = root / 'evaluation/future_vqa'
            directory.mkdir(parents=True)
            reference = {'id': 'future:vqa:1', 'patient': 'p1', 'target': VOCABULARY[0]}
            predictions = directory / 'future_vqa.jsonl'
            predictions.write_text(json.dumps({**reference, 'prediction': reference['target']}) + '\n')
            protocol = scoring_protocol('future_vqa', [reference], cfg)
            protocol_path = directory / 'future_vqa_protocol.json'
            protocol_path.write_text(json.dumps(protocol))
            metrics = score_future_task('future_vqa', [reference], {reference['id']: reference['target']}, protocol=protocol)
            summary = {'checkpoint_sha256': 'checkpoint', 'data_fingerprint': 'data', 'limit': None,
                       'metric_protocol': metric_protocol('table1'), 'split': 'test',
                       'future_data_fingerprint': hashlib.sha256(json.dumps(data_protocol['future'], sort_keys=True).encode()).hexdigest(),
                       'references': {'future_vqa': {'n': 1, 'references_sha256': reference_fingerprint([reference])}},
                       'tasks': {'future_vqa': metrics}, 'predictions_sha256': {'future_vqa': _sha256(predictions)}}
            summary_path = directory / 'summary.json'
            summary_path.write_text(json.dumps(summary))
            self.assertTrue(evaluate_run.completed_job(root, job, 'checkpoint', 'data'))
            protocol_path.unlink()
            with self.assertRaisesRegex(ValueError, 'protocol file is missing'):
                evaluate_run.completed_job(root, job, 'checkpoint', 'data')
            protocol_path.write_text(json.dumps({**protocol, 'schema': 'legacy'}))
            with self.assertRaisesRegex(ValueError, 'current definition'):
                evaluate_run.completed_job(root, job, 'checkpoint', 'data')
            protocol_path.write_text(json.dumps(protocol))
            summary['future_data_fingerprint'] = 'another cohort'
            summary_path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, 'data fingerprint'):
                evaluate_run.completed_job(root, job, 'checkpoint', 'data')
            summary['future_data_fingerprint'] = hashlib.sha256(json.dumps(data_protocol['future'], sort_keys=True).encode()).hexdigest()
            summary['tasks']['future_vqa']['accuracy'] = None
            summary_path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, 'incomplete or nonfinite'):
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
