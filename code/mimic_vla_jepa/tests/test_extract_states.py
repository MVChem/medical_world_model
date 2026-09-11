import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from mimic_vla_jepa.extract_states import validate_cached_state


class StateCacheTests(unittest.TestCase):
    def test_valid_cache_can_be_resumed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.safetensors"
            save_file(
                {
                    "source_state": torch.zeros(256, 2048, dtype=torch.bfloat16),
                    "target_state": torch.ones(256, 2048, dtype=torch.bfloat16),
                },
                path,
            )
            self.assertTrue(validate_cached_state(path))

    def test_partial_or_wrong_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.safetensors"
            save_file(
                {"source_state": torch.zeros(256, 2048, dtype=torch.bfloat16)},
                path,
            )
            self.assertFalse(validate_cached_state(path))
            self.assertFalse(validate_cached_state(Path(directory) / "missing"))


if __name__ == "__main__":
    unittest.main()
