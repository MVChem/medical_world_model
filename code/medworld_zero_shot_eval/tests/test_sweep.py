import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from medworld_zero_shot_eval import sweep


class SweepTest(unittest.TestCase):
    def test_preflights_precede_tests_and_failed_model_is_skipped(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'experiments').mkdir()
            (root / 'experiments/registry.json').write_text('[]')
            cfg = root / 'config.json'
            cfg.write_text(json.dumps({'prepared_data': 'code/data/medworld_0922'}))
            run = root / 'runs/retest'
            calls = []

            def launch(command, **kwargs):
                model = command[command.index('--model') + 1]
                out = Path(command[command.index('--out') + 1])
                resolved = json.loads(Path(command[command.index('--config') + 1]).read_text())
                self.assertEqual(resolved['prepared_data'], str(root / 'code/data/medworld_0922'))
                calls.append((model, out.name))
                out.mkdir(parents=True)
                failed = model == 'qwen4b'
                (out / 'status.json').write_text(json.dumps({'status': 'failed' if failed else 'complete'}))
                return SimpleNamespace(returncode=int(failed))

            with patch('medworld.config.PROJECT', root), patch('sys.argv', [
                'sweep', '--run', str(run), '--config', str(cfg), '--gpus', '6'
            ]), patch.object(sweep.subprocess, 'run', side_effect=launch):
                sweep.main()
            self.assertEqual([phase for _, phase in calls], ['smoke'] * 4 + ['test'] * 3)
            self.assertNotIn(('qwen4b', 'test'), calls)
            status = json.loads((run / 'status.json').read_text())
            self.assertEqual(status['qwen4b']['status'], 'failed')
            self.assertEqual(status['medgemma4b']['status'], 'complete')
            self.assertEqual(json.loads((root / 'experiments/registry.json').read_text()), [])
            self.assertIn('Failed or partial', (root / 'experiments/README.md').read_text())
