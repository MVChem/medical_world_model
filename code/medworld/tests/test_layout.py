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
                       'evaluation.evaluate',
                       'evaluation.compare_run', 'downstream_tasks.text.decoder',
                       'downstream_tasks.common.decoder'):
            with self.subTest(module=module):
                importlib.import_module('medworld.' + module)

    def test_evaluation_uses_one_final_checkpoint(self):
        from medworld.run_experiment import evaluation_jobs
        run = Path("/tmp/example_joint_run")
        jobs = evaluation_jobs(run)
        evaluation = [j for j in jobs if j["module"] == "medworld.evaluation.evaluate"]
        self.assertEqual(len(evaluation), 4)
        self.assertEqual(len({j["id"] for j in jobs}), len(jobs))
        for job in evaluation:
            args = job["args"]
            self.assertEqual(args[args.index("--checkpoint") + 1], str(run / "final.pt"))

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
