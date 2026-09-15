import importlib.util
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
SOURCE = Path(__file__).resolve().parents[1] / 'overnight_guardian.py'
spec = importlib.util.spec_from_file_location('guardian', SOURCE)
guardian = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guardian)


class OrphanGuardTest(unittest.TestCase):
    def test_dead_leader_descendant_is_owned_but_unrelated_process_is_not(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'descendant'
            code = ('import subprocess,sys,time\nfrom pathlib import Path\n'
                    'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(20)"])\n'
                    f'Path({str(path)!r}).write_text(str(p.pid))\n'
                    'time.sleep(.3)\n')
            leader = subprocess.Popen([sys.executable, '-c', code], start_new_session=True)
            outsider = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)'], start_new_session=True)
            record = dict(child_pid=leader.pid, child_start=guardian.proc_info(leader.pid)['start'])
            try:
                leader.wait(timeout=3)
                rows = guardian.members(record, guardian.processes())
                descendant = int(path.read_text())
                self.assertIn(descendant, [row['pid'] for row in rows])
                self.assertNotIn(outsider.pid, [row['pid'] for row in rows])
                for row in rows:
                    guardian.signal_member(row, signal.SIGKILL)
                self.assertIsNone(outsider.poll())
            finally:
                outsider.kill()
                outsider.wait()
                if leader.poll() is None:
                    leader.kill()
                    leader.wait()


if __name__ == '__main__':
    unittest.main()
