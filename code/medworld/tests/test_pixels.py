"""Original CXR pixels and masks use the prepared ROI and do not read arrays."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from medworld.datasets.pixels import human_target, source_canvas


class SourcePixelTests(unittest.TestCase):
    def test_source_and_manual_masks_share_recorded_roi(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "image.png"
            first, second = root / "lung.png", root / "heart.png"
            Image.fromarray(np.full((80, 60), 128, dtype=np.uint8)).save(image)
            Image.fromarray(np.full((80, 60), 255, dtype=np.uint8)).save(first)
            Image.fromarray(np.zeros((80, 60), dtype=np.uint8)).save(second)
            row = {"image": str(image), "box": [0, 64, 512, 384],
                   "masks": [str(first), str(second)]}
            with patch("numpy.load", side_effect=AssertionError("Array cache accessed")):
                source = source_canvas(row)
                targets = human_target(row)
            self.assertEqual(source.shape, (1, 512, 512))
            torch.testing.assert_close(source[:, :, 64:448], torch.full((1, 512, 384), 128 / 255))
            self.assertEqual(source[:, :, :64].sum().item(), 0)
            self.assertEqual(source[:, :, 448:].sum().item(), 0)
            self.assertEqual(targets.shape, (2, 256, 256))
            self.assertEqual(targets[0].sum().item(), 256 * 192)
            self.assertEqual(targets[1].sum().item(), 0)
            self.assertEqual(list(root.glob("*.npy")), [])


if __name__ == "__main__":
    unittest.main()
