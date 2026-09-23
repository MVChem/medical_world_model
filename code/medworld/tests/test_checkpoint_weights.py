"""Saved artifacts retain learned weights and evaluation provenance only."""
from contextlib import redirect_stderr
import importlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

from medworld.config import load_config
from medworld.distributed_train import DistributedTrainer
from medworld.ema import EMATarget
from medworld.model import MedWorld
from medworld.runtime import load_model, read_checkpoint, save_checkpoint


class CompactModel(nn.Module):
    """Exercise the actual compact-state implementation without loading assets."""
    compact_state = MedWorld.compact_state
    restore = MedWorld.restore

    def __init__(self, cfg, device="cpu"):
        super().__init__()
        self.cfg, self.metadata = cfg, {"test": True, "task_initialization_sha256": "a" * 64}
        self.encoder = nn.Module()
        self.encoder.base = nn.Linear(128, 128).requires_grad_(False)
        self.encoder.lora_a = nn.Parameter(torch.randn(3, 128))
        self.encoder.lora_b = nn.Parameter(torch.randn(128, 3))
        self.encoder.slots = nn.Parameter(torch.randn(8, 16))
        self.world = nn.Linear(16, 16)
        self.task_decoder = nn.Linear(16, 16)
        self.classification = nn.Linear(16, 13)
        self.target = EMATarget(self.encoder)
        self.register_buffer("pos_weight", torch.arange(1., 14.))
        self.to(device)


class CheckpointWeightTests(unittest.TestCase):
    def test_online_and_ema_roundtrip_without_base_or_resume_state(self):
        model = CompactModel(load_config())
        with torch.no_grad():
            model.encoder.slots.add_(3)
        model.target.update(model.encoder, .9)
        progress = {"step": 1, "complete": False, "world_size": 2,
                    "offsets": {"classification": 8, "segmentation": 0, "vqa": 0, "temporal": 16},
                    "started_unix": 123., "deadline_unix": 456., "unsaved": lambda: None}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "last.pt"
            with patch("medworld.runtime.rng_state", side_effect=AssertionError("RNG was collected")):
                save_checkpoint(path, model, progress, "data", "base")
            saved = read_checkpoint(path)
            self.assertEqual(set(saved), {"format_version", "config", "metadata", "model", "progress",
                                          "data_fingerprint", "weights_fingerprint"})
            self.assertEqual(saved["progress"], {"step": 1, "complete": False, "world_size": 2,
                             "task_samples": {"classification": 8, "segmentation": 0, "vqa": 0}})
            self.assertFalse(path.with_suffix(".pt.tmp").exists())
            expected = {name for name, value in model.named_parameters() if value.requires_grad}
            self.assertEqual(set(saved["model"]["parameters"]), expected)
            self.assertFalse(any("base" in name for name in saved["model"]["parameters"]))
            self.assertFalse(any("base" in name for name in saved["model"]["ema"]["parameters"]))
            with patch("medworld.model.MedWorld", CompactModel), \
                 patch("medworld.runtime.source_fingerprint", return_value="base"):
                restored, state = load_model(path, "cpu")
            self.assertFalse(restored.training)
            self.assertEqual(state["metadata"], model.metadata)
            self.assertEqual(state["data_fingerprint"], "data")
            for name, value in model.named_parameters():
                if value.requires_grad:
                    torch.testing.assert_close(dict(restored.named_parameters())[name], value, rtol=0, atol=0)
            for name, value in model.target.compact_state()["parameters"].items():
                torch.testing.assert_close(restored.target.compact_state()["parameters"][name], value, rtol=0, atol=0)
            self.assertEqual(int(restored.target.updates), 1)
            torch.testing.assert_close(restored.pos_weight, model.pos_weight, rtol=0, atol=0)
            saved["optimizer"] = {}
            torch.save(saved, path)
            with self.assertRaisesRegex(ValueError, "trained-weights"):
                read_checkpoint(path)

    def test_distributed_save_omits_rank_rng_and_optimizer_payload(self):
        model = CompactModel(load_config())
        with tempfile.TemporaryDirectory() as tmp:
            trainer = DistributedTrainer.__new__(DistributedTrainer)
            trainer.rank, trainer.world_size = 0, 2
            trainer.model, trainer.out = model, Path(tmp)
            trainer.data, trainer.weights_fingerprint = SimpleNamespace(fingerprint="data"), "base"
            trainer.progress = {"step": 0, "complete": False, "world_size": 2,
                                "offsets": {task: 0 for task in ("classification", "segmentation", "vqa", "temporal")}}
            trainer.parameter_spread = Mock()
            with patch("medworld.distributed_train.rank_rng_state", side_effect=AssertionError("RNG was collected")), \
                 patch("medworld.distributed_train.dist.all_gather_object", side_effect=AssertionError("RNG was gathered")), \
                 patch("medworld.distributed_train.dist.barrier") as barrier:
                trainer.save("best.pt")
            trainer.parameter_spread.assert_called_once()
            barrier.assert_called_once()
            self.assertEqual(read_checkpoint(Path(tmp) / "best.pt")["progress"]["world_size"], 2)
            with self.assertRaisesRegex(ValueError, "trained weights only"):
                trainer.resume({})

    def test_resume_entry_points_reject_before_gpu_or_run_mutation(self):
        for name, extra, blocked in (
                ("train", [], "acquire_gpu"),
                ("distributed_train", [], "dist.init_process_group"),
                ("launch_distributed", ["--gpus", "1,2"], "reserve")):
            module = importlib.import_module("medworld." + name)
            with self.subTest(entry=name), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "untouched"
                errors = io.StringIO()
                argv = [name, "--out", str(output), "--resume", str(output / "last.pt"), *extra]
                with patch("sys.argv", argv), redirect_stderr(errors), \
                     patch("medworld." + name + "." + blocked, side_effect=AssertionError("GPU acquired")), \
                     self.assertRaises(SystemExit) as raised:
                    module.main()
                self.assertEqual(raised.exception.code, 2)
                self.assertIn("trained weights only", errors.getvalue())
                self.assertIn("optimizer, RNG and data-stream state are not saved", errors.getvalue())
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
