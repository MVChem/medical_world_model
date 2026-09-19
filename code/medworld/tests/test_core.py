import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from PIL import Image
import torch
from torch import nn

from medworld.config import load_config
from medworld.datasets import patient_holdouts
from medworld.datasets.temporal import TemporalData, directed_pair
from medworld.decoders import ReportDecoder, SpatialHead, validate_state
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
                         {"typo": 1}, {"predictor_width": 31}, {"steps": True}):
            with self.assertRaises(ValueError):
                load_config(overrides=override)


class Tokenizer:
    eos_token_id, pad_token_id = 2, 0

    def apply_chat_template(self, *args, **kwargs):
        return [1, 3]

    def encode(self, text, **kwargs):
        return [3 + ord(c) % 13 for c in text]

    def batch_decode(self, ids, **kwargs):
        return [[int(i) for i in row if i not in (0, 2)] for row in ids]


class CausalLanguage(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.transform = nn.Linear(8, 8)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds=None, input_ids=None, past_key_values=None, **kwargs):
        values = inputs_embeds if inputs_embeds is not None else self.embedding(input_ids)
        cumulative = values.cumsum(1)
        if past_key_values is not None:
            cumulative = cumulative + past_key_values
        return SimpleNamespace(last_hidden_state=self.transform(cumulative.tanh()),
                               past_key_values=cumulative[:, -1:])


class DecoderTests(unittest.TestCase):
    def test_report_ce_reaches_state_with_masked_eos_targets(self):
        language = CausalLanguage().requires_grad_(False)
        head = nn.Linear(8, 16, bias=False).requires_grad_(False)
        decoder = ReportDecoder(language, head, Tokenizer(), load_config(overrides={"report_tokens": 5}))
        state = torch.randn(2, 8, 1024, requires_grad=True)
        loss = decoder.loss(state, ["abcdefg", "a"])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(state.grad.norm()), 0)
        self.assertTrue(all(p.grad is None for p in language.parameters()))
        ids, mask = decoder.targets(["abcdefg", "a"], "cpu")
        self.assertEqual(ids.tolist(), [[9, 10, 11, 12, 2], [9, 2, 0, 0, 0]])
        self.assertEqual(mask.sum(1).tolist(), [5, 2])
        decoder.eval()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.pt"
            torch.save(state.detach().cpu(), path)
            loaded = torch.load(path, weights_only=True)
            self.assertEqual(decoder.generate(loaded, 4), decoder.generate(state, 4))
        with self.assertRaises(ValueError):
            decoder.generate(state, 0)
        with self.assertRaises(ValueError):
            validate_state(torch.zeros(1, 4, 1024))

    def test_per_sample_eos_and_cache_positions(self):
        class ScriptedLanguage(CausalLanguage):
            def __init__(self):
                super().__init__()
                self.calls = []

            def forward(self, **kwargs):
                i = len(self.calls)
                self.calls.append(kwargs)
                hidden = torch.zeros(2, 1, 8)
                hidden[0, 0, 2] = 10  # first sample ends immediately
                hidden[1, 0, 4 if i < 2 else 2] = 10
                return SimpleNamespace(last_hidden_state=hidden, past_key_values="cache")
        language = ScriptedLanguage()
        head = nn.Linear(8, 16, bias=False)
        with torch.no_grad():
            head.weight.zero_()
            head.weight[:8].copy_(torch.eye(8))
        decoder = ReportDecoder(language, head, Tokenizer(), load_config()).eval()
        result = decoder.generate(torch.randn(2, 8, 1024), 8)
        self.assertEqual(result, [[], [4, 4]])
        self.assertEqual(len(language.calls), 3)
        self.assertEqual(language.calls[1]["position_ids"].tolist(), [[10], [10]])
        self.assertEqual(language.calls[2]["position_ids"].tolist(), [[11], [11]])
        self.assertEqual(language.calls[2]["input_ids"].tolist(), [[0], [4]])

    def test_spatial_loss_reads_only_visual_slots(self):
        state = torch.randn(1, 8, 1024, requires_grad=True)
        prediction = SpatialHead("segmentation")(torch.rand(1, 1, 32, 32), state[:, 4:])
        self.assertEqual(prediction.shape, (1, 3, 256, 256))
        prediction.square().mean().backward()
        self.assertEqual(float(state.grad[:, :4].abs().sum()), 0)
        self.assertGreater(float(state.grad[:, 4:].abs().sum()), 0)


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
                             "report": text, "labels": [0] * 6} for ident, text in (("a", "before"), ("b", "after"))]
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
            self.assertEqual(batch["report_targets"], ["before"])
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
