"""Update budgets and multi-shard provenance for the Qwen9B forecast upgrade."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import pretrained_weight_signature, stage1_budget_reached
from common import digest


class TrainingBudgetTest(unittest.TestCase):
    def test_fixed_stage1_updates_override_elapsed_time(self):
        cfg = dict(stage1_hours=1.5, max_stage1_steps=1694)
        self.assertFalse(stage1_budget_reached(cfg, 1693, 100000))
        self.assertTrue(stage1_budget_reached(cfg, 1694, 1))
        self.assertTrue(stage1_budget_reached(dict(stage1_hours=1.5), 1, 5400))

    def test_shard_change_changes_signature_and_missing_shard_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = root / 'model.safetensors.index.json'
            index.write_text(json.dumps({'weight_map': {'a': 'part1.safetensors', 'b': 'part2.safetensors'}}))
            (root / 'part1.safetensors').write_bytes(b'part1')
            (root / 'part2.safetensors').write_bytes(b'part2')
            before = pretrained_weight_signature(root)
            self.assertEqual(len(before['shards']), 2)
            (root / 'part2.safetensors').write_bytes(b'changed')
            self.assertNotEqual(before, pretrained_weight_signature(root))
            (root / 'part2.safetensors').unlink()
            with self.assertRaises(FileNotFoundError):
                pretrained_weight_signature(root)

    def test_legacy_single_shard_preserves_resume_signature(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'model.safetensors-00001-of-00001.safetensors'
            path.write_bytes(b'legacy')
            self.assertEqual(pretrained_weight_signature(root), digest(path))
            (root / 'model.safetensors.index.json').write_text(json.dumps(
                {'weight_map': {'a': path.name, 'b': path.name}}))
            self.assertEqual(pretrained_weight_signature(root), digest(path))


if __name__ == '__main__':
    unittest.main()
