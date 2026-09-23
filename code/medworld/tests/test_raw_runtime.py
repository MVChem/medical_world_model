"""Task-only runs must not request slot assets, temporal batches, or EMA updates."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from medworld import FORMAT_VERSION
from medworld.architecture import RAW_INPUT, architecture, require_reviewed_data, uses_slot_branch, uses_temporal
from medworld.batching import BatchPrefetch
from medworld.config import load_config
from medworld.run_experiment import arm_config
from medworld.runtime import Trainer, read_checkpoint, seed_all, source_fingerprint


class RawData:
    fingerprint = "fixed-current-task-data"

    def __init__(self):
        self.requests = []

    def training_batch(self, task, offset, batch_size, seed):
        if task == "temporal":
            raise AssertionError("Task-only training requested temporal data")
        self.requests.append((task, offset, batch_size))
        return {"x": torch.full((batch_size, 3), (offset % 11 + 1) / 11)}

    def rows(self, *args):
        return [0]

    def batch(self, *args):
        return {"x": torch.ones(1, 3)}


class RawTaskModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg, self.metadata = cfg, {"test": True}
        self.readout = nn.Linear(3, 1)
        self.target = None

    def forward(self, task, batch, temporal_batch=None):
        if temporal_batch is not None:
            raise AssertionError("Task-only forward received temporal data")
        loss = self.readout(batch["x"]).square().mean()
        return loss, {task: loss.detach()}

    def compact_state(self):
        return {"params": self.state_dict(), "ema": None}

    def restore(self, state):
        if state["ema"] is not None:
            raise AssertionError("Task-only checkpoint contains EMA")
        self.load_state_dict(state["params"])


class RawRuntimeTests(unittest.TestCase):
    def test_raw_training_requires_reviewed_data_even_when_evaluation_disabled(self):
        for slots in (False, True):
            cfg = load_config(overrides={"slot_conditioning": slots, "testing": {"enabled": False}})
            for current in (SimpleNamespace(), SimpleNamespace(metadata={"schema": "medworld-prepared-v1"})):
                with self.subTest(slots=slots, current=current), self.assertRaisesRegex(ValueError, "medworld-prepared-v2"):
                    require_reviewed_data(cfg, SimpleNamespace(current=current))
            require_reviewed_data(cfg, SimpleNamespace(current=SimpleNamespace(metadata={"schema": "medworld-prepared-v2"})))

    def test_new_configs_default_to_raw_inputs_and_reject_other_architectures(self):
        self.assertEqual(architecture(load_config()), RAW_INPUT)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"slot_conditioning": False, "visual_consistency_weight": .1}))
            cfg = load_config(path)
            self.assertEqual(architecture(cfg), RAW_INPUT)
            self.assertFalse(uses_slot_branch(cfg))
            self.assertEqual(cfg["visual_consistency_weight"], 0)
            self.assertFalse(uses_temporal(cfg))
        for value in ("legacy_v3", "typo", None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "architecture"):
                load_config(overrides={"architecture": value})

    def test_checkpoint_requires_current_format_and_explicit_architecture(self):
        state = {"format_version": FORMAT_VERSION, "config": load_config(),
                 "metadata": {}, "model": {}, "progress": {},
                 "data_fingerprint": "data", "weights_fingerprint": "weights"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            torch.save(state, path)
            self.assertEqual(read_checkpoint(path)["config"]["architecture"], RAW_INPUT)
            for version, cfg in ((3, load_config()), (FORMAT_VERSION, {}),
                                 (FORMAT_VERSION, {"architecture": "legacy_v3"})):
                with self.subTest(version=version, config=cfg):
                    torch.save({**state, "format_version": version, "config": cfg}, path)
                    with self.assertRaises(ValueError):
                        read_checkpoint(path)

    def test_raw_baseline_normalizes_auxiliary_weights_in_both_entry_points(self):
        slots = load_config(overrides={"visual_consistency_weight": .1, "latent_weight": 2})
        for baseline in (arm_config(slots, False), load_config(overrides={
                "slot_conditioning": False, "visual_consistency_weight": .1, "latent_weight": 2})):
            self.assertFalse(uses_slot_branch(baseline))
            self.assertFalse(uses_temporal(baseline))
            self.assertEqual((baseline["latent_weight"], baseline["visual_consistency_weight"]), (0, 0))
        self.assertEqual((slots["latent_weight"], slots["visual_consistency_weight"]), (2, .1))
        self.assertFalse(uses_temporal(load_config(overrides={"latent_weight": 0})))

    def test_baseline_source_fingerprint_never_reads_pretrained_assets(self):
        cfg = load_config(overrides={"slot_conditioning": False, "qwen": "/absent/qwen", "jepa": "/absent/jepa"})
        with patch("medworld.runtime._sha256", side_effect=AssertionError("Read pretrained file")), \
             patch.object(Path, "iterdir", side_effect=AssertionError("Listed pretrained assets")):
            first = source_fingerprint(cfg)
            second = source_fingerprint({**cfg, "qwen": "/different/qwen", "jepa": "/different/jepa"})
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)

    def test_distributed_prefetch_has_only_current_task_requests_and_offsets(self):
        cfg = load_config(overrides={"slot_conditioning": False, "batch_size": 3, "accumulation": 2})
        progress = {"step": 0, "offsets": {task: 0 for task in ("classification", "segmentation", "vqa", "temporal")}}
        data = RawData()
        stream = BatchPrefetch(data, cfg, progress, rank=1, world_size=2)
        try:
            task, batches, marker = stream.next()
            self.assertEqual(task, "classification")
            self.assertTrue(all(temporal is None for _, temporal in batches))
            self.assertEqual(marker["offsets"]["classification"], 12)
            self.assertEqual(marker["offsets"]["temporal"], 0)
            self.assertTrue(all(value == 0 for value in progress["offsets"].values()))
        finally:
            stream.close()
        self.assertTrue(data.requests)
        self.assertTrue(all(task != "temporal" for task, _, _ in data.requests))

    def test_task_only_training_and_weight_restoration_need_no_encoder_or_ema(self):
        cfg = load_config(overrides={"slot_conditioning": False, "steps": 6,
                                     "batch_size": 2, "accumulation": 2, "validation_samples": 1})
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            def make(name):
                seed_all(19)
                directory = Path(tmp) / name
                directory.mkdir()
                return Trainer(RawTaskModel(cfg), RawData(), directory, source_fingerprint(cfg))
            full, partial = make("full"), make("partial")
            self.assertTrue(full.run())
            step = partial.optimizer.step
            def stop_after_step(*args, **kwargs):
                result = step(*args, **kwargs)
                partial.request_stop()
                return result
            partial.optimizer.step = stop_after_step
            self.assertFalse(partial.run())
            saved = read_checkpoint(partial.out / "last.pt")
            self.assertIsNone(saved["model"]["ema"])
            restored = make("restored")
            restored.model.restore(saved["model"])
            for name, value in partial.model.state_dict().items():
                torch.testing.assert_close(value, restored.model.state_dict()[name], rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, "trained weights only"):
                restored.resume(saved)
            self.assertNotIn("optimizer", saved)
            self.assertNotIn("rng", saved)
            self.assertEqual(full.progress["offsets"]["temporal"], 0)
            records = [json.loads(line) for line in (full.out / "metrics.jsonl").read_text().splitlines()]
            self.assertTrue(all(row["ema_updates"] == row["temporal_global_batch"] == 0 for row in records))
            self.assertTrue(all(row["current_global_batch"] == 4 for row in records))


if __name__ == "__main__":
    unittest.main()
