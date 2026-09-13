"""Scientific extraction contracts: layer identity, pooling, and SR source isolation."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from frozen_slots_extract import (block_indices, branch_image, cache_array, pool_slot,
                                  required_branches, tensor_hash)


class FrozenSlotsTests(unittest.TestCase):
    def test_quarter_depth_including_final(self):
        self.assertEqual(block_indices(12), [2, 5, 8, 11])
        self.assertEqual(block_indices(24), [5, 11, 17, 23])
        self.assertEqual(block_indices(27), [6, 13, 20, 26])
        with self.assertRaises(ValueError):
            block_indices(3)

    def test_native_mean_and_fixed_channel_alignment(self):
        # Token permutation preserves a global slot, and width-768 models retain
        # all their native channels with explicitly empty trailing channels.
        h = torch.arange(5 * 768).reshape(5, 768).float()
        expected = h.mean(0)
        out = pool_slot(h)
        torch.testing.assert_close(out[:768], expected)
        self.assertEqual(out[768:].count_nonzero().item(), 0)
        torch.testing.assert_close(pool_slot(h.flip(0)), out)
        wider = torch.arange(1152).float()[None].expand(7, -1)
        indices = torch.floor((torch.arange(1024) + .5) * 1152 / 1024).long()
        torch.testing.assert_close(pool_slot(wider), indices.float())
        with self.assertRaises(ValueError):
            pool_slot(torch.ones(2, 5, 768))
        with self.assertRaises(ValueError):
            pool_slot(torch.full((5, 768), float("nan")))

    def test_sr_branch_never_indexes_hr_source(self):
        class ForbiddenHR:
            def __getitem__(self, _index):
                raise AssertionError("SR accessed HR pixels")

        lr = np.full((1, 128, 128), 73, dtype=np.uint8)
        image = branch_image("lr", 0, ForbiddenHR(), lr)
        self.assertEqual(image.size, (128, 128))
        self.assertEqual(image.getpixel((0, 0)), (73, 73, 73))
        with self.assertRaises(ValueError):
            branch_image("lr", 0, ForbiddenHR(), np.zeros((1, 512, 512), dtype=np.uint8))
        rows = [{"tasks": ["segmentation", "sr"]}, {"tasks": ["segmentation"]}, {"tasks": []}]
        np.testing.assert_array_equal(required_branches(rows), [[True, True], [True, False], [False, False]])

    def test_missing_cache_and_changed_shape_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "slots.npy"
            a = cache_array(path, (2, 4, 1024))
            self.assertTrue(np.isnan(a).all())
            a[0] = 3
            a.flush()
            resumed = cache_array(path, (2, 4, 1024))
            self.assertTrue((resumed[0] == 3).all())
            self.assertTrue(np.isnan(resumed[1]).all())
            with self.assertRaises(ValueError):
                cache_array(path, (3, 4, 1024))

    def test_hash_compares_backbones_independently_of_merger(self):
        one = {"blocks.0.weight": torch.ones(3, 3, dtype=torch.bfloat16),
               "merger.weight": torch.zeros(3, 4, dtype=torch.bfloat16)}
        two = dict(one, **{"merger.weight": torch.zeros(3, 7, dtype=torch.bfloat16)})
        self.assertNotEqual(tensor_hash(one), tensor_hash(two))
        self.assertEqual(tensor_hash(one, True), tensor_hash(two, True))
        two["blocks.0.weight"] = torch.zeros(3, 3, dtype=torch.bfloat16)
        self.assertNotEqual(tensor_hash(one, True), tensor_hash(two, True))


if __name__ == "__main__":
    unittest.main()
