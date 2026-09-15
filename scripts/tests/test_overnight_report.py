import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'overnight_report.py'
spec = importlib.util.spec_from_file_location('overnight_report', SOURCE)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class ResultGatingTest(unittest.TestCase):
    def test_separate_direction_cohort_is_not_inserted_in_main_table(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw = root / 'code/medworld_baselines/runs/raw_models_20260911/qwen08b/test'
            raw.mkdir(parents=True)
            (raw / 'metrics.json').write_text(json.dumps({'table1': {'n': 297, 'ap': .7}}))
            direction = root / 'code/medworld_dense_baselines/runs/dense_20260912/qwen08b'
            direction.mkdir(parents=True)
            (direction / 'direction_metrics.json').write_text(json.dumps({'complete': True, 'macro_f1': .8}))
            out = root / 'out'
            report.report(root, out)
            import csv
            rows = list(csv.DictReader((out / 'table1.csv').read_text(encoding='utf-8-sig').splitlines()))
            self.assertEqual(rows[0]['AP'], '0.7')
            self.assertEqual(rows[0]['Direction F1'], '')
            self.assertIn('0.8000', (out / 'REPORT.md').read_text())

    def test_existing_metrics_without_queue_receipt_remain_pending(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run = root / report.RUNS['joint']
            run.mkdir(parents=True)
            artifact = run / 'metrics.json'
            artifact.write_text(json.dumps({'dice': .99}))
            mapping = [{'table': 2, 'method': 'unverified', 'protocol': 'test',
                        'artifacts': [{'path': str(artifact), 'metrics': {'Dice pseudo': 'dice'}}]}]
            (run / 'table_export.json').write_text(json.dumps(mapping))
            out = root / 'out'
            report.report(root, out)
            import csv
            row = list(csv.DictReader((out / 'table2.csv').read_text(encoding='utf-8-sig').splitlines()))[-1]
            self.assertEqual(row['Status'], 'pending')
            self.assertEqual(row['Dice pseudo'], '')


if __name__ == '__main__':
    unittest.main()
