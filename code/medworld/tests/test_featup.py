"""Geometry, teacher isolation, and unified spatial gradient/checkpoint contracts."""
import io
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from medworld.adaptation import LoRALinear
from medworld.config import load_config
from medworld.featup import (FeatUpSpatialHead, FrozenFeatureTeacher, consistency_loss,
                            feature_loss, patch_grid)
from medworld.model import MedWorld


class TinyVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=64, spatial_merge_size=2)
        self.blocks = nn.ModuleList([nn.Linear(64, 64) for _ in range(4)])

    def forward(self, hidden_states, grid_thw):
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return hidden_states


class PixelTeacher(nn.Module):
    def forward(self, pixels, processor, vision_pixels):
        return torch.nn.functional.adaptive_avg_pool2d(pixels, (4, 4)).expand(-1, 64, -1, -1).detach()


class SpatialHarness(MedWorld):
    def __init__(self):
        nn.Module.__init__(self)
        self.cfg = load_config(overrides={"spatial_decoder": "featup"})
        self.encoder = nn.Module()
        self.encoder.slot_queries = nn.Parameter(torch.randn(4, 1024))
        self.segmentation, self.sr = FeatUpSpatialHead("segmentation"), FeatUpSpatialHead("sr")
        self.featup_teacher, self.processor = PixelTeacher(), None
        self.register_buffer("pos_weight", torch.ones(13))
        self.target = None

    def encode(self, images, **kwargs):
        return self.encoder.slot_queries[None].expand(len(images), -1, -1)

    def temporal_loss(self, batch):
        loss = self.encoder.slot_queries.square().mean()
        return loss, {"temporal": loss.detach()}


class FeatUpTests(unittest.TestCase):
    def test_geometry_and_masked_consistency(self):
        # Qwen 2x2 merge-block order, not ordinary raster order.
        tokens = torch.arange(16.).view(1, 16, 1)
        grid = patch_grid(tokens, 4, 4)
        self.assertEqual(grid[0, 0].tolist(), [[0, 1, 4, 5], [2, 3, 6, 7],
                                             [8, 9, 12, 13], [10, 11, 14, 15]])
        field = torch.randn(1, 64, 8, 8, requires_grad=True)
        target = field.detach().clone().requires_grad_()
        theta = torch.tensor([[[[1., 0, 0], [0, 1., 0]]]])
        loss = feature_loss(field, target[:, None], theta, torch.ones(1, 1, 8, 8))
        self.assertLess(float(loss.detach()), 1e-10)
        loss.backward()
        self.assertIsNone(target.grad)
        self.assertEqual(float(feature_loss(field, target[:, None], theta, torch.zeros(1, 1, 8, 8)).detach()), 0)

    def test_fixed_teacher_unchanged_by_online_adaptation(self):
        vision = TinyVision().requires_grad_(False)
        teacher = FrozenFeatureTeacher(vision)
        self.assertIs(teacher.vision.blocks[0].weight, vision.blocks[0].weight)
        inputs = torch.randn(16, 64)
        before = teacher.vision(inputs, None).clone()
        vision.blocks[0] = LoRALinear(vision.blocks[0], 4, 8)
        with torch.no_grad():
            vision.blocks[0].lora_b.fill_(1)
        teacher.train()
        self.assertFalse(teacher.vision.training)
        self.assertTrue(all(not p.requires_grad for p in teacher.parameters()))
        torch.testing.assert_close(teacher.vision(inputs, None), before, rtol=0, atol=0)
        self.assertFalse(torch.allclose(vision(inputs, None), before))

    def test_stage1_and_replay_gradients_and_compact_roundtrip(self):
        model = SpatialHarness()
        for task, size, channels in (("segmentation", 256, 3), ("sr", 512, 1)):
            batch = {"images": [None], "pixels": torch.rand(1, 1, 128, 128),
                     "targets": torch.rand(1, channels, size, size),
                     "mask": torch.ones(1, 1, size, size), "ids": ["case"]}
            for replay in (False, True):
                model.zero_grad(set_to_none=True)
                loss, parts = model("temporal", {}, task, batch) if replay else model(task, batch)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                head = getattr(model, task)
                for p in (head.upsample.kernel[-1].weight, head.feature.weight, model.encoder.slot_queries):
                    self.assertIsNotNone(p.grad)
                    self.assertGreater(float(p.grad.norm()), 0)
                prefix = "replay_" if replay else ""
                feature = parts[prefix + task + "_feature"]
                weighted = parts[prefix + task + "_feature_weighted"]
                torch.testing.assert_close(weighted, feature * (.0001 if task == "sr" else .1))
        with torch.no_grad():
            expected = model.sr(batch["pixels"], model.encode([None]))
            buffer = io.BytesIO()
            torch.save(model.compact_state(), buffer)
            buffer.seek(0)
            restored = SpatialHarness()
            restored.restore(torch.load(buffer, weights_only=True))
            torch.testing.assert_close(restored.sr(batch["pixels"], restored.encode([None])), expected, rtol=0, atol=0)
        # Decoder inference does not request targets or the feature teacher.
        self.assertEqual(tuple(expected.shape), (1, 1, 512, 512))

    def test_config_validation(self):
        for values in ({"spatial_decoder": "typo"}, {"featup_views": 0},
                       {"featup_feature_weight": -1}, {"featup_sr_scale": float("nan")}):
            with self.assertRaises(ValueError):
                load_config(overrides=values)



class PixelFallbackTests(unittest.TestCase):
    def test_source_only_ignores_pixel_caches_and_preserves_lr_input(self):
        import tempfile
        from pathlib import Path
        import numpy as np
        from PIL import Image
        from medworld.datasets.current import MultiTaskData
        from medworld.datasets.pixels import source_canvas, low_resolution, human_target
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / 'source.png'
            Image.fromarray(np.random.default_rng(7).integers(0, 256, (80, 60), dtype=np.uint8)).save(image)
            mask = root / 'mask.png'
            Image.fromarray(np.full((80, 60), 255, dtype=np.uint8)).save(mask)
            source = {'image': str(image), 'box': [0, 64, 512, 384], 'masks': [str(mask), str(mask)]}
            data = MultiTaskData.__new__(MultiTaskData)
            data.old, data.dense = root / 'old', root / 'dense'
            data.old.mkdir(); data.dense.mkdir()
            data._old_count = data._dense_count = 1
            data._old_sources = data._dense_sources = [source]
            data._arrays = {'pseudo': np.ones((1, 3, 256, 256), dtype=np.float16)}
            hr = (source_canvas(source)[0] * 255).round().byte().numpy()
            lr = (low_resolution(source_canvas(source))[0] * 255).round().byte().numpy()
            human = human_target(source).byte().numpy()
            paths = [(data.old / 'images.npy', hr), (data.dense / 'images.npy', hr),
                     (data.dense / 'lr_images.npy', lr), (data.dense / 'human_masks.npy', human)]
            for path, array in paths:
                np.save(path, array[None])
            row = {'id': 'case', 'subject_id': 'patient', 'image_index': 0, 'index': 0,
                   'old_index': 0, 'human_index': 0, 'box': source['box'], 'kind': 'montgomery',
                   'report_target': 'reference', 'labels': [0] * 13}
            from unittest.mock import patch
            with patch('numpy.load', side_effect=AssertionError('Pixel cache accessed')):
                cached = {task: data._example(task, 'test', row) for task in
                          ('classification', 'report', 'segmentation', 'sr')}
            torch.testing.assert_close(cached['sr']['targets'][0], torch.from_numpy(hr).float() / 255)
            torch.testing.assert_close(cached['segmentation']['targets'], torch.from_numpy(human).float())
            data._arrays = {'pseudo': data._arrays['pseudo']}
            for path, _ in paths:
                path.unlink()
            for task, expected in cached.items():
                actual = data._example(task, 'test', row)
                for key in ('pixels', 'targets', 'mask'):
                    if key in expected:
                        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
                np.testing.assert_array_equal(actual['image'], expected['image'])
            sr = data._example('sr', 'test', row)
            self.assertEqual(sr['image'].size, (128, 128))
            np.testing.assert_array_equal(np.asarray(sr['image'])[:, :, 0], lr)
            self.assertFalse(any(data.dense.glob('*.npy')))


if __name__ == "__main__":
    unittest.main()
