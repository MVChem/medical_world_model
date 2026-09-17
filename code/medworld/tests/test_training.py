from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from medworld.config import load_config
from medworld.ema import EMATarget
from medworld.runtime import Trainer, read_checkpoint, seed_all


class ToyModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg, self.metadata = cfg, {"test": True}
        self.encoder = nn.Sequential(nn.Linear(3, 3), nn.Dropout(.1))
        self.readout = nn.Linear(3, 1)
        self.target = None

    def begin_stage2(self):
        self.target = EMATarget(self.encoder)

    def current_loss(self, task, batch):
        loss = self.readout(self.encoder(batch["x"])).square().mean()
        return loss, {task: loss.detach()}

    def temporal_loss(self, batch):
        output = self.encoder(batch["x"])
        loss = (output - self.target(batch["x"] + .25)).square().mean() + self.readout(output).square().mean()
        return loss, {"temporal": loss.detach()}

    def compact_state(self):
        return {"params": {n: p.detach().clone() for n, p in self.named_parameters() if p.requires_grad},
                "ema": None if self.target is None else self.target.compact_state()}

    @torch.no_grad()
    def restore(self, state):
        parameters = dict(self.named_parameters())
        for name, value in state["params"].items():
            parameters[name].copy_(value)
        if state["ema"] is not None:
            if self.target is None:
                self.begin_stage2()
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
    def test_resume_matches_uninterrupted_with_accumulation_and_replay(self):
        cfg = load_config(overrides={"stage1_steps": 4, "stage2_steps": 4, "stage1_accumulation": 2,
                                     "stage2_accumulation": 2, "replay_every": 2, "validation_samples": 1})
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            def make(name):
                seed_all(81)
                path = Path(tmp) / name
                path.mkdir()
                return Trainer(ToyModel(cfg), ToyData(), path, "fixed-base")
            full = make("full")
            self.assertTrue(full.run_stage())
            stage1_readout = full.model.readout.weight.clone()
            full.enter_stage2()
            torch.testing.assert_close(full.model.readout.weight, stage1_readout)
            full.run_stage()
            partial = make("partial")
            partial.run_stage()
            partial.enter_stage2()
            step = partial.optimizer.step
            calls = []
            def stop_after_two(*args, **kwargs):
                result = step(*args, **kwargs)
                calls.append(1)
                if len(calls) == 2:
                    partial.request_stop()
                return result
            partial.optimizer.step = stop_after_two
            self.assertFalse(partial.run_stage())
            self.assertEqual(int(partial.model.target.updates), 2)
            saved = read_checkpoint(partial.out / "last.pt")
            resumed = make("resumed")
            resumed.resume(saved)
            self.assertTrue(resumed.run_stage())
            self.assertEqual(resumed.progress, full.progress)
            self.assertEqual(int(resumed.model.target.updates), 4)  # not 8 microbatches or 12 with replay
            for name, parameter in full.model.named_parameters():
                torch.testing.assert_close(parameter, dict(resumed.model.named_parameters())[name], rtol=0, atol=0)
            self.assertEqual(resumed.progress["replay_index"], 2)
            self.assertEqual(resumed.progress["offsets"]["temporal"], 8)
            saved["data_fingerprint"] = "changed"
            with self.assertRaises(ValueError):
                resumed.resume(saved)

    def test_stage2_requires_completed_stage1(self):
        cfg = load_config()
        trainer = Trainer(ToyModel(cfg), ToyData(), "/unused", "fixed")
        with self.assertRaises(ValueError):
            trainer.enter_stage2()


if __name__ == "__main__":
    unittest.main()
