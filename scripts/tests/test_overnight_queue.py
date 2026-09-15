import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'overnight_queue.py'
spec = importlib.util.spec_from_file_location('overnight_queue', SOURCE)
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


class DeadlineQueueTest(unittest.TestCase):
    def test_rejects_dependency_cycle(self):
        jobs = [dict(id='a', argv=['python'], artifacts=['a.json'], deps=['b']),
                dict(id='b', argv=['python'], artifacts=['b.json'], deps=['a'])]
        with self.assertRaisesRegex(ValueError, 'cycle'):
            queue.validate_plan(dict(jobs=jobs))

    def test_partial_or_wrong_budget_is_not_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'metrics.json'
            job = dict(artifacts=[str(path)], success_fields={str(path): {'epochs': 20}})
            path.write_text(json.dumps(dict(status='partial', epochs=20)))
            self.assertIsNone(queue.valid_artifacts(job))
            path.write_text(json.dumps(dict(epochs=19)))
            self.assertIsNone(queue.valid_artifacts(job))
            path.write_text(json.dumps(dict(epochs=20)))
            self.assertEqual(set(queue.valid_artifacts(job)), {str(path)})

    def test_worker_hard_deadline_without_coordinator(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            start = dt.datetime.now(dt.timezone.utc)
            marker = root / 'metrics.json'
            child_script = root / 'child.py'
            child_script.write_text('import signal,time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\ntime.sleep(30)\n')
            record = dict(job=dict(id='deadline', argv=[sys.executable, str(child_script)], artifacts=[str(marker)]),
                          cards=[], project=folder, attempt_id='test', result=str(root / 'result.json'),
                          soft_deadline=(start + dt.timedelta(seconds=1)).isoformat(),
                          hard_deadline=(start + dt.timedelta(seconds=2)).isoformat())
            queue.atomic(root / 'record.json', record)
            before = time.monotonic()
            subprocess.run([sys.executable, str(SOURCE), 'worker', '--record', str(root / 'record.json')],
                           check=True, timeout=8)
            result = queue.read(root / 'result.json')
            self.assertLess(time.monotonic() - before, 7)
            self.assertEqual(result['returncode'], -9)
            self.assertEqual(result['reason'], 'hard deadline')
            saved = queue.read(root / 'record.json')
            self.assertIsNone(queue.identity(saved['child_pid']))

    def test_binary_checkpoint_and_complete_status_are_valid(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / 'checkpoint_final.pt'
            checkpoint.write_bytes(b'PK\x03\x04\x80\xff\x00checkpoint')
            status = Path(folder) / 'status.json'
            status.write_text(json.dumps(dict(state='complete', stage_step=2400)))
            job = dict(artifacts=[str(checkpoint), str(status)],
                       success_fields={str(status): {'state': 'complete', 'stage_step': 2400}})
            self.assertEqual(queue.valid_artifacts(job),
                             {str(checkpoint): queue.sha(checkpoint), str(status): queue.sha(status)})

    def test_wrong_pid_identity_does_not_signal(self):
        self.assertFalse(queue.signal_owned(os.getpid(), 'different-start-time', 15))

    def test_success_receipt_requires_matching_attempt(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'result.json'
            path.write_text(json.dumps(dict(attempt_id='stale', returncode=0, finished=time.time())))
            job = dict(status='running', worker_pid=999999999, worker_start='1', result=str(path),
                       record=str(Path(folder) / 'missing.json'), attempt_id='current')
            queue.reconcile(job)
            self.assertEqual(job['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
