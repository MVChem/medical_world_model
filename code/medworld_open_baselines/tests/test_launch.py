"""Check queue snapshots without starting training or touching GPUs."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('baseline_launch', Path(__file__).resolve().parents[1] / 'launch.py')
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


class LaunchTests(unittest.TestCase):
    def test_snapshot_has_executor_and_protocol_without_old_scripts_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ['medworld_open_baselines', 'medworld_table1', 'medworld_common',
                         'medworld_stage1', 'medworld_baselines', 'medworld_dense_baselines']:
                (root / 'code' / name).mkdir(parents=True)
            executor = root / 'code/medworld_common/overnight_queue.py'
            executor.write_text('# shared executor\n')
            protocol = root / 'code/medworld_open_baselines/swinir_protocol'
            protocol.mkdir()
            (protocol / 'config.py').write_text('seed = 1\n')
            plan = root / 'plan.json'
            plan.write_text(json.dumps({'jobs': []}))
            run = root / 'runs/test_20260918'
            argv = ['launch.py', '--project', str(root), '--run', str(run), '--name', 'test', '--plan', str(plan)]
            with patch.object(launch.sys, 'argv', argv), patch.object(launch.subprocess, 'Popen', return_value=SimpleNamespace(pid=42)) as start, contextlib.redirect_stdout(io.StringIO()):
                launch.main()
            source = run / 'scheduler_test/source/code/medworld_open_baselines'
            self.assertEqual((source / 'executor.py').read_text(), executor.read_text())
            self.assertEqual((source / 'swinir_protocol/config.py').read_text(), 'seed = 1\n')
            command = start.call_args.args[0]
            self.assertEqual(command[command.index('--gpu-order') + 1], '1,2,3,6,7,0')
            self.assertEqual(command[command.index('--max-gpus') + 1], '6')


if __name__ == '__main__':
    unittest.main()
