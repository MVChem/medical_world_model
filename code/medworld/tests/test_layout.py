"""Public package boundaries and configurable manifest path stability."""
import importlib
from pathlib import Path
import tempfile
import unittest
from medworld.config import load_config


class LayoutTests(unittest.TestCase):
    def test_runtime_entrypoints_import(self):
        for module in ('datasets.current', 'datasets.temporal', 'datasets.protocol',
                       'datasets.unified', 'downstream_tasks.segmentation.decoder',
                       'downstream_tasks.super_resolution.decoder', 'evaluation.evaluate',
                       'evaluation.baseline_audit', 'evaluation.compare_run',
                       'evaluation.clinical_report', 'evaluation.dense_reference'):
            with self.subTest(module=module):
                importlib.import_module('medworld.' + module)

    def test_data_alias_does_not_change_protocol_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'actual').mkdir()
            (root / 'alias').symlink_to(root / 'actual', target_is_directory=True)
            cfg = load_config(overrides={'dense_data': 'alias'}, root=root)
            self.assertEqual(cfg['dense_data'], str(root / 'alias'))
            self.assertTrue(Path(cfg['dense_data']).is_dir())


if __name__ == '__main__':
    unittest.main()
