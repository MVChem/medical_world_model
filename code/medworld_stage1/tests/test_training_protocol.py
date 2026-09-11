"""Contracts that must survive training-loop refactors."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bootstrap
import torch
from training import optimizer_for, should_stop
from training_variants import gradient_record, resolve_variant
from snapshot import create_snapshot, source_hashes


class TrainingProtocolTests(unittest.TestCase):
    def test_baseline_follows_updates_even_after_reference_deadline(self):
        cfg = dict(
            max_steps=24000,
            max_train_hours=1,
            train_deadline="2020-01-01T00:00:00+00:00",
        )
        self.assertFalse(
            should_stop(cfg, "noslots", Path("reference"), 3176, 99999, 9999999999)
        )
        self.assertTrue(should_stop(cfg, "noslots", Path("reference"), 24000, 0, 0))
        self.assertTrue(should_stop(cfg, "slot44", None, 3176, 99999, 9999999999))
        self.assertFalse(
            should_stop(cfg, "legacy", Path("reference"), 3176, 99999, 9999999999)
        )

    def test_variant_cannot_silently_change_task_definition(self):
        self.assertEqual(resolve_variant({"slots": 8}), "legacy")
        self.assertEqual(resolve_variant({"slots": 8, "qa_manifest": "qa"}), "slot44")
        self.assertEqual(resolve_variant({"slots": 0}), "noslots")
        with self.assertRaises(ValueError):
            resolve_variant({"slots": 0}, "slot44")

    def test_optimizer_preserves_lora_groups_and_frozen_weights(self):
        model = torch.nn.Module()
        model.register_parameter("lora_A", torch.nn.Parameter(torch.ones(2)))
        model.register_parameter("head", torch.nn.Parameter(torch.ones(2)))
        model.register_parameter(
            "frozen", torch.nn.Parameter(torch.ones(2), requires_grad=False)
        )
        optimizer = optimizer_for(
            model, dict(lora_learning_rate=5e-5, learning_rate=1e-4)
        )
        self.assertEqual([g["lr"] for g in optimizer.param_groups], [5e-5, 1e-4])
        self.assertEqual(optimizer.defaults["betas"], (0.9, 0.95))
        self.assertEqual(optimizer.defaults["weight_decay"], 0.01)
        self.assertEqual(
            {id(p) for g in optimizer.param_groups for p in g["params"]},
            {id(model.lora_A), id(model.head)},
        )

    def test_wrong_slot_route_and_padded_context_are_rejected(self):
        encoder = torch.nn.Module()
        encoder.register_parameter("slots", torch.nn.Parameter(torch.ones(8, 3)))
        encoder.slots.grad = torch.ones_like(encoder.slots)
        state = torch.ones(1, 8, 3, requires_grad=True)
        state.grad = torch.ones_like(state)  # Classification must not read slots 5–8.
        model = SimpleNamespace(encoder=encoder, last_state=state)
        with self.assertRaisesRegex(RuntimeError, "wrong state group"):
            gradient_record(model, "classification", 0, False, "slot44")
        state.grad[:, 4:] = 0
        self.assertEqual(
            gradient_record(model, "classification", 0, False, "slot44")[
                "state_route_grad_norms"
            ][4:],
            [0.0] * 4,
        )
        values = torch.ones(1, 3, 3, requires_grad=True)
        values.grad = torch.ones_like(values)
        model.last_memory = SimpleNamespace(
            values=values, mask=torch.tensor([[1, 1, 0]])
        )
        with self.assertRaisesRegex(RuntimeError, "padded token"):
            gradient_record(model, "classification", 0, False, "noslots")

    def test_snapshot_contains_shared_code_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            source = create_snapshot(run)
            self.assertIn("medworld_common/qwen.py", source_hashes(source))
            self.assertIn("training.py", source_hashes(source))
            self.assertIn(str(bootstrap.PROJECT), (source / "bootstrap.py").read_text())
            with self.assertRaises(FileExistsError):
                create_snapshot(run)


if __name__ == "__main__":
    unittest.main()
