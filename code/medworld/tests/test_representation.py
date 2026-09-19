import unittest
import torch
from torch import nn
from medworld.config import load_config
from medworld.representation import SlotSpatialReconstruction
from medworld.representation.targets import feature_loss, patch_grid


class RepresentationTests(unittest.TestCase):
    def test_slots_are_required_and_receive_spatial_gradients(self):
        torch.manual_seed(11)
        head = SlotSpatialReconstruction()
        slots = torch.randn(2, 4, 1024, requires_grad=True)
        field = head(slots)
        self.assertEqual(field.shape, (2, 64, 64, 64))
        target = torch.randn(2, 1, 64, 16, 16)
        theta = torch.tensor([[1.,0.,0.],[0.,1.,0.]])[None,None].expand(2,1,2,3)
        loss = feature_loss(field, target, theta, torch.ones(2,1,128,128))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue((slots.grad.norm(dim=-1) > 0).all())
        self.assertFalse(torch.allclose(field.detach(), head(torch.zeros_like(slots))))
        self.assertFalse(torch.allclose(field.detach()[0], field.detach()[1]))
        with self.assertRaises(ValueError):head(torch.randn(2,8,1024))

    def test_roundtrip_has_no_frozen_target_or_input_pixels(self):
        head = SlotSpatialReconstruction()
        clone = SlotSpatialReconstruction()
        clone.load_state_dict(head.state_dict())
        slots = torch.randn(1,4,1024)
        torch.testing.assert_close(head(slots),clone(slots))
        self.assertEqual(list(__import__('inspect').signature(head.forward).parameters), ['slots'])

    def test_default_off_and_validation(self):
        self.assertEqual(load_config()['visual_consistency_weight'],0)
        for cfg in ({'visual_consistency_weight':-1},{'visual_consistency_weight':float('nan')},
                    {'visual_consistency_views':0}):
            with self.assertRaises(ValueError):load_config(overrides=cfg)

    def test_patch_grid_restores_spatial_merge_order(self):
        grid = torch.arange(16).reshape(4,4)
        tokens = grid.reshape(2,2,2,2).permute(0,2,1,3).reshape(1,16,1)
        torch.testing.assert_close(patch_grid(tokens,4,4)[0,0],grid)


if __name__ == '__main__':unittest.main()
