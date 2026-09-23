import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from medworld.config import load_config
from medworld.datasets.protocol import _sha256
from medworld.evaluation.compare_run import compare
from medworld.evaluation.protocol import metric_protocol
from medworld.evaluation.protocol import FUTURE_TASKS
from medworld.evaluation.selection import reference_manifest, select_vqa


class ComparisonTests(unittest.TestCase):
    def raw_pair(self, root):
        checkpoints = []
        for name, slots, score in [('baseline', False, .3), ('slots', True, .4)]:
            run = root / name
            directory = run / 'evaluation/vqa'
            directory.mkdir(parents=True)
            cfg = load_config(overrides={'slot_conditioning': slots, 'visual_consistency_weight': .1,
                                         'testing': {'tasks': ['vqa']}})
            (run / 'config.json').write_text(json.dumps(cfg))
            (run / 'final.pt').write_bytes(name.encode())
            checkpoints.append({'config': cfg, 'metadata': {'task_initialization_sha256': 'a' * 64},
                                'progress': {'step': 10, 'complete': True,
                                             'task_samples': {'classification': 4, 'segmentation': 3, 'vqa': 3}},
                                'data_fingerprint': 'same', 'weights_fingerprint': name + '-sources'})
            record = {'id': 'x', 'patient': '1', 'question': 'q', 'answer': ['yes'], 'semantic_type': 'verify'}
            _, selection = select_vqa([record])
            (directory / 'vqa.jsonl').write_text(json.dumps(record) + '\n')
            (directory / 'summary.json').write_text(json.dumps({'metric_protocol': metric_protocol('table2'),
                'references': {'vqa': reference_manifest('vqa', [record])}, 'vqa_selection': selection,
                'limit': None, 'split': 'test',
                'data_fingerprint': 'same', 'checkpoint_sha256': _sha256(run / 'final.pt'),
                'predictions_sha256': {'vqa': _sha256(directory / 'vqa.jsonl')},
                'tasks': {'vqa': {'n': 1, 'exact_match': score, 'micro_f1': score}}}))
        return checkpoints

    def test_raw_pair_allows_branch_specific_sources_and_records_objectives(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoints = self.raw_pair(root)
            with patch('medworld.runtime.read_checkpoint', side_effect=checkpoints):
                rows = compare(root / 'baseline', root / 'slots', root / 'comparison')
            self.assertAlmostEqual(rows[0]['delta'], .1)
            result = json.loads((root / 'comparison/comparison.json').read_text())
            self.assertEqual(result['architecture'], 'raw_input_v1')
            self.assertEqual(result['arms']['baseline']['latent_weight'], 0)
            self.assertEqual(result['arms']['baseline']['visual_consistency_weight'], 0)
            self.assertFalse(result['arms']['baseline']['slot_branch'])
            self.assertTrue(result['arms']['slots']['slot_branch'])
            self.assertIn('Raw-input task-only baseline', (root / 'comparison/COMPARISON.md').read_text())

    def test_old_metric_protocol_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoints = self.raw_pair(root)
            path = root / 'baseline/evaluation/vqa/summary.json'
            summary = json.loads(path.read_text())
            summary.pop('metric_protocol')
            path.write_text(json.dumps(summary))
            with patch('medworld.runtime.read_checkpoint', side_effect=checkpoints), self.assertRaises(ValueError):
                compare(root / 'baseline', root / 'slots', root / 'comparison')

    def test_future_training_counts_are_checked_before_current_task_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoints = self.raw_pair(root)
            for name, checkpoint in zip(('baseline', 'slots'), checkpoints):
                checkpoint['config']['future_enabled'] = True
                checkpoint['progress']['task_samples'].update({task: 4 for task in FUTURE_TASKS})
                (root / name / 'config.json').write_text(json.dumps(checkpoint['config']))
            with patch('medworld.runtime.read_checkpoint', side_effect=checkpoints):
                self.assertEqual(len(compare(root / 'baseline', root / 'slots', root / 'comparison')), 2)
            checkpoints[1]['progress']['task_samples']['future_report'] = 8
            with patch('medworld.runtime.read_checkpoint', side_effect=checkpoints), self.assertRaisesRegex(ValueError, 'sample counts'):
                compare(root / 'baseline', root / 'slots', root / 'comparison')

    def test_raw_pair_rejects_missing_or_stale_prediction_hashes(self):
        for tamper in (False, True):
            with self.subTest(tamper=tamper), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                checkpoints = self.raw_pair(root)
                if tamper:
                    path = root / 'slots/evaluation/vqa/vqa.jsonl'
                    path.write_text(path.read_text().replace('"q"', '"changed question"'))
                else:
                    path = root / 'slots/evaluation/vqa/summary.json'
                    summary = json.loads(path.read_text())
                    summary.pop('predictions_sha256')
                    path.write_text(json.dumps(summary))
                with patch('medworld.runtime.read_checkpoint', side_effect=checkpoints), self.assertRaisesRegex(ValueError, 'Prediction file SHA256'):
                    compare(root / 'baseline', root / 'slots', root / 'comparison')

    def test_raw_pair_rejects_unmatched_initialization_or_auxiliary_baseline(self):
        cases = [
            ('initialization', lambda a, b: b['metadata'].update(task_initialization_sha256='b' * 64)),
            ('initialization', lambda a, b: a['metadata'].clear()),
            ('auxiliary', lambda a, b: a['config'].update(visual_consistency_weight=.1)),
            ('auxiliary', lambda a, b: a['config'].update(latent_weight=1)),
            ('Unsupported MedWorld architecture', lambda a, b: b['config'].update(architecture='legacy_v3')),
            ('sample counts', lambda a, b: b['progress']['task_samples'].update(vqa=4)),
            ('sample counts', lambda a, b: b['progress'].pop('task_samples')),
            ('completed training', lambda a, b: b['progress'].pop('complete')),
            ('equal configs', lambda a, b: b['config'].update(batch_size=7)),
        ]
        for message, change in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                checkpoints = self.raw_pair(root)
                change(*checkpoints)
                with patch('medworld.runtime.read_checkpoint', side_effect=checkpoints), self.assertRaisesRegex(ValueError, message):
                    compare(root / 'baseline', root / 'slots', root / 'comparison')

    def test_only_configured_tasks_are_compared_and_mismatches_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoints = self.raw_pair(root)
            with patch('medworld.runtime.read_checkpoint', side_effect=checkpoints):
                rows = compare(root / 'baseline', root / 'slots', root / 'comparison')
            self.assertEqual(len(rows), 2)
            self.assertAlmostEqual(rows[0]['delta'], .1)
            checkpoints[1]['progress']['step'] = 11
            with patch('medworld.runtime.read_checkpoint', side_effect=checkpoints), self.assertRaisesRegex(ValueError, 'update counts'):
                compare(root / 'baseline', root / 'slots', root / 'comparison')
            for saved in checkpoints:
                saved['config']['total_hours'] = 1.5
            with patch('medworld.runtime.read_checkpoint', side_effect=checkpoints), self.assertRaisesRegex(ValueError, 'update counts'):
                compare(root / 'baseline', root / 'slots', root / 'comparison')
            for saved in checkpoints:
                saved['config']['total_hours'] = 0
            checkpoints[1]['progress']['step'] = 10
            (root / 'slots/evaluation/vqa/vqa.jsonl').write_text('{"id":"other","patient":"1","question":"q","answer":["yes"],"semantic_type":"verify"}\n')
            summary_path = root / 'slots/evaluation/vqa/summary.json'
            summary = json.loads(summary_path.read_text())
            summary['predictions_sha256']['vqa'] = _sha256(root / 'slots/evaluation/vqa/vqa.jsonl')
            summary_path.write_text(json.dumps(summary))
            with patch('medworld.runtime.read_checkpoint', side_effect=checkpoints), self.assertRaisesRegex(ValueError, 'IDs or references'):
                compare(root / 'baseline', root / 'slots', root / 'comparison')
