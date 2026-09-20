import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from medworld.config import load_config
from medworld.datasets.protocol import _sha256
from medworld.evaluation.compare_run import compare


class ComparisonTests(unittest.TestCase):
    def test_only_configured_tasks_are_compared_and_mismatches_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoints = []
            for name, slots, score in [('baseline', False, .3), ('slots', True, .4)]:
                run = root / name
                directory = run / 'evaluation/vqa'
                directory.mkdir(parents=True)
                cfg = load_config(overrides={'slot_conditioning': slots, 'testing': {'tasks': ['vqa']}})
                (run / 'config.json').write_text(json.dumps(cfg))
                (run / 'final.pt').write_bytes(name.encode())
                checkpoints.append({'config': cfg, 'progress': {'step': 10}, 'data_fingerprint': 'same', 'weights_fingerprint': 'same'})
                (directory / 'vqa.jsonl').write_text(json.dumps({'id': 'x', 'patient': '1', 'question': 'q', 'answer': ['yes']}) + '\n')
                (directory / 'summary.json').write_text(json.dumps({'limit': None, 'split': 'test',
                    'data_fingerprint': 'same', 'checkpoint_sha256': _sha256(run / 'final.pt'),
                    'tasks': {'vqa': {'n': 1, 'exact_match': score, 'micro_f1': score}}}))
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
            (root / 'slots/evaluation/vqa/vqa.jsonl').write_text('{"id":"other","patient":"1","question":"q","answer":["yes"]}\n')
            with patch('medworld.runtime.read_checkpoint', side_effect=checkpoints), self.assertRaisesRegex(ValueError, 'IDs or references'):
                compare(root / 'baseline', root / 'slots', root / 'comparison')
