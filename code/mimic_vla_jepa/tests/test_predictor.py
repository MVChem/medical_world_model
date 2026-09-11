import tempfile
import unittest
from pathlib import Path

import torch

from mimic_vla_jepa.backbones import local_hf_revision, query_positions
from mimic_vla_jepa.predictor import PredictorConfig, VLAJEPAPredictor


class PredictorTests(unittest.TestCase):
    def test_local_hf_revision_reads_only_valid_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata = (
                Path(directory)
                / ".cache"
                / "huggingface"
                / "download"
                / "config.json.metadata"
            )
            metadata.parent.mkdir(parents=True)
            revision = "a" * 40
            metadata.write_text(f"{revision}\netag\n123\n", encoding="utf-8")
            self.assertEqual(local_hf_revision(directory), revision)
            metadata.write_text("not-a-revision\n", encoding="utf-8")
            self.assertIsNone(local_hf_revision(directory))

    def test_query_positions_are_exact(self):
        input_ids = torch.tensor([[1, 9, 2, 9, 3]])
        positions = query_positions(input_ids, token_id=9, expected=2)
        self.assertEqual(positions.tolist(), [1, 3])
        with self.assertRaisesRegex(ValueError, "expected 3"):
            query_positions(input_ids, token_id=9, expected=3)

    def test_small_predictor_forward_and_backward(self):
        config = PredictorConfig(
            image_size=32,
            patch_size=16,
            state_dim=64,
            action_dim=32,
            query_tokens=2,
            predictor_dim=96,
            depth=1,
            heads=4,
        )
        model = VLAJEPAPredictor(config)
        source = torch.randn(2, 4, 64)
        query = torch.randn(2, 2, 32)
        target = torch.randn(2, 4, 64)
        output = model(source, query)
        self.assertEqual(tuple(output.shape), tuple(target.shape))
        loss = torch.nn.functional.l1_loss(output, target)
        loss.backward()
        self.assertTrue(
            any(parameter.grad is not None for parameter in model.parameters())
        )

    def test_shape_mismatch_is_rejected(self):
        config = PredictorConfig(
            image_size=32,
            patch_size=16,
            state_dim=64,
            action_dim=32,
            query_tokens=2,
            predictor_dim=96,
            depth=1,
            heads=4,
        )
        model = VLAJEPAPredictor(config)
        with self.assertRaisesRegex(ValueError, "source_state"):
            model(torch.randn(1, 5, 64), torch.randn(1, 2, 32))


if __name__ == "__main__":
    unittest.main()
