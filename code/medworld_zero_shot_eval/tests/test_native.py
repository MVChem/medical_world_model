import unittest
import torch
from medworld_zero_shot_eval.evaluate import positive_probability, yes_no_ids


class NativeClassificationTest(unittest.TestCase):
    def test_probability_normalizes_only_yes_no_without_argmax(self):
        logits = torch.tensor([100., 2., 0., -10.])
        self.assertAlmostEqual(float(positive_probability(logits, [1, 2])),
                               float(torch.sigmoid(torch.tensor(2.))), places=6)
        self.assertAlmostEqual(float(positive_probability(logits, [2, 1])),
                               1-float(torch.sigmoid(torch.tensor(2.))), places=6)

    def test_reject_multitoken_candidates(self):
        class Tokenizer:
            def encode(self, text, **kwargs):
                return [1, 2] if text == 'Yes' else [3]
        with self.assertRaises(ValueError):
            yes_no_ids(Tokenizer())
