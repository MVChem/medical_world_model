"""CPU checks for attached slot gradients, adapter updates, and LR-only inputs."""
import unittest
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medworld_joint.model import EightSlotHead, LoRALinear, JointModel, adapter_state, load_adapter_state
from medworld_joint.train import OnlineCorpus
from frozen_slots_train import objective


class JointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_dense_losses_reach_every_slot(self):
        for task, size in (("segmentation", 256), ("sr", 128)):
            torch.manual_seed(11)
            head = EightSlotHead(task)
            slots = torch.randn(2, 8, 1024, requires_grad=True)
            image = torch.rand(2, 1, size, size)
            prediction = head(image, slots)
            target = torch.rand_like(prediction)
            mask = torch.ones_like(prediction[:, :1])
            loss = objective(task, prediction, target, mask)
            loss.backward()
            self.assertIsNotNone(slots.grad)
            self.assertTrue((slots.grad.norm(dim=2) > 0).all())
            self.assertTrue(torch.isfinite(slots.grad).all())

    def test_lora_changes_effective_backbone_but_preserves_base_weights(self):
        torch.manual_seed(8)
        module = LoRALinear(nn.Linear(6, 5), rank=2, alpha=4)
        original = module.base.weight.detach().clone()
        x = torch.randn(4, 6, requires_grad=True)
        before = module(x).detach().clone()
        optimizer = torch.optim.SGD([p for p in module.parameters() if p.requires_grad], lr=.1)
        module(x).square().mean().backward()
        self.assertGreater(float(module.lora_b.grad.norm()), 0)
        self.assertGreater(float(x.grad.norm()), 0)
        self.assertIsNone(module.base.weight.grad)
        optimizer.step()
        self.assertTrue(torch.equal(original, module.base.weight))
        self.assertFalse(torch.equal(before, module(x)))

    def test_full_token_readout_keeps_all_context_gradients(self):
        head = EightSlotHead("segmentation", full_tokens=True)
        context = torch.randn(1, 41, 1024, requires_grad=True)
        head(torch.rand(1, 1, 256, 256), context).square().mean().backward()
        self.assertTrue((context.grad.norm(dim=2) > 0).all())

    def test_adapter_checkpoint_restores_trainables_exactly(self):
        torch.manual_seed(4)
        model = JointModel("qwen08b", "sr", "image_only", device="cpu")
        state = adapter_state(model)
        with torch.no_grad():
            next(model.parameters()).add_(1)
        load_adapter_state(model, state)
        for name, param in model.named_parameters():
            self.assertTrue(torch.equal(param.detach(), state[name]))

    def test_sr_online_donor_uses_lr_pixels_only(self):
        corpus = OnlineCorpus.__new__(OnlineCorpus)
        corpus.device = torch.device("cpu")
        corpus.task = "sr"
        corpus.condition = "shuffled_slots"
        corpus.images = np.full((2, 512, 512), 230, dtype=np.uint8)
        corpus.lr_images = np.stack([np.full((128, 128), value, dtype=np.uint8) for value in (11, 37)])
        corpus.rows = [dict(box=[0, 0, 512, 512]) for _ in range(2)]
        corpus.donors = {0: 1, 1: 0}
        corpus.slots = None
        visual, sources, target, _ = corpus.batch([0])
        self.assertEqual(sources[0].size, (128, 128))
        self.assertTrue((np.asarray(sources[0]) == 37).all())
        self.assertAlmostEqual(float(visual.mean()), 11 / 255, places=6)
        self.assertAlmostEqual(float(target.mean()), 230 / 255, places=6)


if __name__ == "__main__":
    unittest.main()
