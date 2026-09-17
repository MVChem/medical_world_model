import unittest

import torch
from torch import nn
import torch.nn.functional as F

from medworld.decoders import spatial_loss
from medworld_spatial.geometry import feature_loss, patch_grid, semantic_loss, view
from medworld_spatial.model import SlotReader, SparseSpatialModel


def fixture(variant="featup_semantic"):
    torch.manual_seed(11)
    reader = SlotReader(nn.ModuleList([nn.Sequential(nn.LayerNorm(12), nn.Linear(12, 1024)) for _ in range(4)]),
                        torch.randn(4, 1024) * .02, nn.LayerNorm(1024))
    model = SparseSpatialModel(reader, torch.randn(4, 32), variant)
    batch = {"raw": torch.randn(2, 4, 16, 12), "fusion": torch.randn(2, 4, 1024), "grid": (4, 4),
             "pixels": torch.rand(2, 1, 128, 128), "valid": torch.ones(2, 1, 256, 256)}
    return model, batch


class GeometryTests(unittest.TestCase):
    def test_merge_block_order_is_not_naive_reshape(self):
        tokens = torch.tensor([0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15])
        actual = patch_grid(tokens[None, :, None], 4, 4, 2)
        torch.testing.assert_close(actual[0, 0], torch.arange(16).reshape(4, 4))
        with self.assertRaises(ValueError):
            patch_grid(tokens[None, :, None], 3, 4, 2)

    def test_feature_teacher_is_fixed_and_transform_matches(self):
        field = torch.randn(2, 8, 16, 16, requires_grad=True)
        theta = torch.tensor([[[1., 0, 0], [0, 1., 0]], [[.8, 0, .1], [0, .8, -.1]]])[None].repeat(2, 1, 1, 1)
        targets = torch.stack([F.adaptive_avg_pool2d(view(field.detach(), theta[:, j]), (4, 4)) for j in range(2)], 1)
        targets.requires_grad_(True)
        loss = feature_loss(field, targets, theta, torch.ones(2, 1, 32, 32))
        self.assertLess(float(loss.detach()), 1e-10)
        altered = feature_loss(field + .1, targets, theta, torch.ones(2, 1, 32, 32))
        altered.backward()
        self.assertGreater(float(field.grad.norm()), 0)
        self.assertIsNone(targets.grad)

    def test_semantic_soft_targets_and_confidence_filter(self):
        logits = torch.randn(2, 4, 8, 8, requires_grad=True)
        attention = logits.flatten(2).softmax(-1).reshape_as(logits)
        target = torch.full((2, 4, 2, 2), .7)
        zero, coverage = semantic_loss(attention, target, torch.ones(2, 1, 32, 32))
        self.assertEqual(float(zero.detach()), 0)
        self.assertEqual(float(coverage), 0)
        target[:, :, 0, 0] = .95
        loss, coverage = semantic_loss(attention, target, torch.ones(2, 1, 32, 32))
        loss.backward()
        self.assertEqual(float(coverage), 1)
        self.assertGreater(float(logits.grad.norm()), 0)
        self.assertTrue(torch.isfinite(logits.grad).all())


class ModelTests(unittest.TestCase):
    def test_attention_sums_masks_and_slot_gradients(self):
        model, batch = fixture()
        batch["valid"][:, :, :, :32] = 0
        result = model(batch, "segmentation")
        self.assertEqual(result["prediction"].shape, (2, 3, 256, 256))
        for name in ("encoder_attention", "semantic_attention"):
            weights = result[name]
            torch.testing.assert_close(weights.sum((-1, -2)), torch.ones(weights.shape[:2]), atol=1e-6, rtol=1e-6)
        for name in ("decoder_attention32", "decoder_attention64", "local_attention"):
            weights = result[name]
            torch.testing.assert_close(weights.sum(1), torch.ones_like(weights[:, 0]), atol=1e-6, rtol=1e-6)
        self.assertEqual(float(result["semantic_attention"][..., :8].detach().sum()), 0)
        target = torch.rand_like(result["prediction"])
        spatial_loss("segmentation", result["prediction"], target, batch["valid"]).backward()
        self.assertGreater(float(model.reader.queries.grad.norm()), 0)
        self.assertGreater(float(model.semantic.query.weight.grad.norm()), 0)

    def test_image_only_invariance_and_real_slot_sensitivity(self):
        model, batch = fixture("image_only")
        model.eval()
        with torch.no_grad():
            before = model(batch, "segmentation")["prediction"]
            changed = dict(batch, raw=batch["raw"] + torch.randn_like(batch["raw"]), fusion=batch["fusion"] * -2)
            torch.testing.assert_close(before, model(changed, "segmentation")["prediction"], rtol=0, atol=0)
            model.variant = "slots"
            result = model(batch, "segmentation")
            replaced = model(batch, "segmentation", replacement=torch.zeros_like(result["slots"]))
            self.assertGreater(float((result["prediction"] - replaced["prediction"]).abs().max()), 1e-6)

    def test_sr_shape_and_targets_are_not_forward_inputs(self):
        model, batch = fixture()
        batch["targets"] = torch.rand(2, 1, 512, 512)
        batch["valid"] = torch.ones_like(batch["targets"])
        with torch.no_grad():
            before = model(batch, "sr")["prediction"]
            self.assertEqual(before.shape, batch["targets"].shape)
            batch["targets"] = torch.zeros_like(batch["targets"])
            torch.testing.assert_close(before, model(batch, "sr")["prediction"], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
