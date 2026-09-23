from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from medworld.config import load_config
from medworld.ema import EMATarget
from medworld.model import MedWorld
from medworld.runtime import Trainer, read_checkpoint, seed_all


class ToyModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg, self.metadata = cfg, {"test": True}
        self.encoder = nn.Sequential(nn.Linear(3, 3), nn.Dropout(.1))
        self.readout = nn.Linear(3, 1)
        self.world = nn.Linear(3, 3)
        self.target = EMATarget(self.encoder)

    forward = MedWorld.forward

    @property
    def device(self):
        return self.encoder[0].weight.device

    def current_loss(self, task, batch):
        loss = self.readout(self.encoder(batch["x"])).square().mean()
        return loss, {task: loss.detach()}

    def temporal_loss(self, batch):
        output = self.world(self.encoder(batch["x"]))
        loss = (output - self.target(batch["x"] + .25)).square().mean() + self.readout(output).square().mean()
        return loss, {"temporal": loss.detach()}

    def compact_state(self):
        return {"params": {n: p.detach().clone() for n, p in self.named_parameters() if p.requires_grad},
                "ema": self.target.compact_state()}

    @torch.no_grad()
    def restore(self, state):
        parameters = dict(self.named_parameters())
        for name, value in state["params"].items():
            parameters[name].copy_(value)
        self.target.restore(state["ema"])


class ToyData:
    fingerprint = "synthetic-fixed-data"

    def rows(self, *args):
        return [0]

    def batch(self, *args):
        return {"x": torch.ones(1, 3)}

    def training_batch(self, task, offset, batch_size, seed):
        return {"x": torch.full((batch_size, 3), (offset % 11 + 1) / 11)}


class TrainingTests(unittest.TestCase):
    def test_joint_training_saves_weights_and_ema_with_accumulation(self):
        cfg = load_config(overrides={"steps": 6, "accumulation": 2, "batch_size": 3,
                                     "task_batch_sizes": {"temporal": 4}, "validation_samples": 1})
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            def make(name):
                seed_all(81)
                path = Path(tmp) / name
                path.mkdir()
                return Trainer(ToyModel(cfg), ToyData(), path, "fixed-base")
            full = make("full")
            self.assertTrue(full.run())
            partial = make("partial")
            step = partial.optimizer.step
            calls = []
            def stop_after_two(*args, **kwargs):
                result = step(*args, **kwargs)
                calls.append(1)
                if len(calls) == 2:
                    partial.request_stop()
                return result
            partial.optimizer.step = stop_after_two
            self.assertFalse(partial.run())
            self.assertEqual(int(partial.model.target.updates), 2)
            saved = read_checkpoint(partial.out / "last.pt")
            restored = make("restored")
            restored.model.restore(saved["model"])
            for name, parameter in partial.model.named_parameters():
                torch.testing.assert_close(parameter, dict(restored.model.named_parameters())[name], rtol=0, atol=0)
            self.assertEqual(int(restored.model.target.updates), 2)
            self.assertEqual(saved["progress"], {"step": 2, "complete": False, "world_size": 1,
                             "task_samples": {"classification": 6, "segmentation": 6, "vqa": 0}})
            with self.assertRaisesRegex(ValueError, "trained weights only"):
                restored.resume(saved)
            self.assertEqual(int(full.model.target.updates), 6)  # one EMA per optimizer update
            self.assertEqual(full.progress["offsets"]["temporal"], 48)
            for task in ("classification", "segmentation", "vqa"):
                self.assertEqual(full.progress["offsets"][task], 12)
            final = read_checkpoint(full.out / "final.pt")
            self.assertEqual(final["progress"]["step"], 6)
            self.assertTrue(final["progress"]["complete"])
            self.assertNotIn("optimizer", final)
            self.assertNotIn("rng", final)

    def test_first_update_combines_both_objectives_and_freezes_target(self):
        model = ToyModel(load_config()).eval()
        current = {"x": torch.ones(2, 3)}
        temporal = {"x": torch.full((3, 3), .5)}
        loss, parts = model("classification", current, temporal)
        expected = model.current_loss("classification", current)[0] + model.temporal_loss(temporal)[0]
        torch.testing.assert_close(loss, expected)
        loss.backward()
        for parameter in (model.encoder[0].weight, model.readout.weight, model.world.weight):
            self.assertGreater(float(parameter.grad.norm()), 0)
        self.assertTrue(all(p.grad is None for p in model.target.parameters()))
        self.assertIn("temporal_temporal", parts)
        self.assertEqual(int(model.target.updates), 0)

    def test_checkpoint_rejects_old_training_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.pt"
            torch.save({"format_version": 1}, path)
            with self.assertRaises(ValueError):
                read_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
