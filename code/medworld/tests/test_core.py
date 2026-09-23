import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import torch
from torch import nn

from medworld.config import load_config
from medworld.datasets import patient_holdouts
from medworld.datasets.temporal import TemporalData, directed_pair
from medworld.ema import EMATarget
from medworld.predictor import WorldModel, time_features


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(3, 3).requires_grad_(False)
        self.adapter = nn.Linear(3, 3)
        self.drop = nn.Dropout(.5)
        self.register_buffer("counter", torch.tensor(0))

    def forward(self, x):
        return self.adapter(self.drop(self.base(x)))


class EMATests(unittest.TestCase):
    def test_mutable_copy_frozen_alias_and_equation(self):
        online = TinyEncoder()
        target = EMATarget(online)
        self.assertIs(online.base.weight, target.encoder.base.weight)
        self.assertIsNot(online.adapter.weight, target.encoder.adapter.weight)
        self.assertIsNot(online.counter, target.encoder.counter)
        before = target.encoder.adapter.weight.clone()
        with torch.no_grad():
            online.adapter.weight.add_(2)
            online.counter.add_(3)
        target.update(online, .75)
        torch.testing.assert_close(target.encoder.adapter.weight, before + .5)
        self.assertEqual(int(target.encoder.counter), 3)
        self.assertEqual(int(target.updates), 1)
        target.train(True)
        self.assertFalse(target.training)
        self.assertFalse(target.encoder.drop.training)
        self.assertFalse(target(torch.ones(1, 3)).requires_grad)
        online(torch.ones(1, 3)).sum().backward()
        self.assertIsNotNone(online.adapter.weight.grad)
        self.assertTrue(all(p.grad is None for p in target.parameters()))

    def test_roundtrip_and_guards(self):
        online = TinyEncoder()
        target = EMATarget(online)
        target.update(online, .9)
        other = EMATarget(online)
        other.restore(target.compact_state())
        self.assertEqual(int(other.updates), 1)
        for a, b in zip(target.parameters(), other.parameters()):
            torch.testing.assert_close(a, b)
        for momentum in (-.1, 1, float("nan")):
            with self.assertRaises(ValueError):
                target.update(online, momentum)
        state = target.compact_state()
        state["parameters"].pop("adapter.weight")
        with self.assertRaises(ValueError):
            other.restore(state)
        online.adapter.weight.requires_grad_(False)
        with self.assertRaises(ValueError):
            target.update(online, .9)


class PredictorTests(unittest.TestCase):
    def test_signed_actual_time_and_gradient(self):
        features = time_features(torch.tensor([72., -72., 0.]))
        torch.testing.assert_close(features[0, 1], features[1, 1])
        torch.testing.assert_close(features[0, [0, 2]], -features[1, [0, 2]])
        torch.testing.assert_close(features[2], torch.zeros(3))
        world = WorldModel(load_config(overrides={"predictor_width": 32, "predictor_depth": 1}))
        state = torch.randn(1, 8, 1024, requires_grad=True)
        initial = world(state, torch.tensor([72.]))
        torch.testing.assert_close(initial, torch.nn.functional.layer_norm(state, (1024,)))
        initial.square().mean().backward()
        self.assertGreater(float(world.output.weight.grad.norm()), 0)
        self.assertIsNotNone(state.grad)
        # The zero-initialized residual intentionally starts direction-neutral.
        nn.init.normal_(world.output.weight, std=.01)
        self.assertFalse(torch.allclose(world(state, torch.tensor([72.])), world(state, torch.tensor([-72.]))))
        with self.assertRaises(ValueError):
            world(state, torch.tensor([float("nan")]))
        with self.assertRaises(ValueError):
            world(state, torch.tensor([72]))

    def test_configuration_rejects_silent_protocol_changes(self):
        for override in ({"report_weight": 0}, {"ema_momentum": 1}, {"bidirectional": "false"},
                         {"typo": 1}, {"predictor_width": 31}, {"steps": True},
                         {"cpu_threads": 0}, {"require_fast_kernels": "true"}):
            with self.assertRaises(ValueError):
                load_config(overrides=override)


class DataTests(unittest.TestCase):
    def test_global_holdout_precedence(self):
        current = {"report": {"train": [{"subject_id": "p1"}, {"subject_id": "p2"}],
                              "test": [{"subject_id": "p3"}]}}
        temporal = [{"patient": "p1", "split": "test"}, {"patient": "p2", "split": "validate"},
                    {"patient": "p3", "split": "train"}]
        result = patient_holdouts(current, temporal)
        self.assertEqual(result, {"p1": "test", "p2": "validate", "p3": "test"})

    def test_actual_interval_reversal_and_source_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (4, 8)).save(root / "image.png")
            observations = [{"id": ident, "patient": "p", "split": "train", "image": "image.png",
                             "report": text, "labels": [0] * 13} for ident, text in (("a", "before"), ("b", "after"))]
            pair = {"id": "pair", "patient": "p", "split": "train", "source": "a", "target": "b",
                    "horizon": 1, "realized_gap_hours": 27.5}
            for name, rows in (("observations", observations), ("train", [pair]), ("validate", []), ("test", [])):
                (root / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            data = TemporalData(root)
            forward, backward = data.directed("train")
            self.assertEqual((forward["delta_hours"], backward["delta_hours"]), (27.5, -27.5))
            self.assertEqual(backward["source"], "b")
            self.assertEqual(backward["target"], "a")
            batch = data.batch([backward])
            self.assertEqual(batch["source"]["texts"], ["after"])
            self.assertEqual(batch["target"]["texts"], ["before"])
            del data.lookup["b"]  # inaccessible target must not affect forward inference
            source = data.batch([forward], source_only=True)
            self.assertEqual(set(source), {"source", "delta_hours"})
            self.assertEqual(source["source"]["texts"], ["before"])
            with self.assertRaises(KeyError):
                data.batch([forward])
            for gap in (0, -1, float("inf")):
                with self.assertRaises(ValueError):
                    directed_pair(dict(pair, realized_gap_hours=gap))


if __name__ == "__main__":
    unittest.main()
