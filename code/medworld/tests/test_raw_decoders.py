"""Independent raw paths, causal answer prediction, and padding boundaries."""
import json
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from medworld.downstream_tasks.common.decoder import ByteTokenizer, TaskDecoder
from medworld.downstream_tasks.text.decoder import TextDecoder


def tiny_config(**overrides):
    return {"decoder_width": 16, "decoder_depth": 2, "vision_pixels": 16,
            "task_patch_size": 8, "context_tokens": 48, "answer_tokens": 24,
            "generation_tokens": 24, **overrides}


class RawDecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(17)

    def test_utf8_roundtrip_and_serialized_targets_end_at_eos(self):
        tokenizer = ByteTokenizer()
        serialized = json.dumps(["肺", "yes"], ensure_ascii=False)
        self.assertEqual(tokenizer.decode(tokenizer.encode(serialized)), serialized)
        self.assertEqual(tokenizer.encode("肺", max_bytes=2), [])
        decoder = TextDecoder(tiny_config(answer_tokens=7))
        ids, valid = decoder.targets(["abcdefghi", "肺"], "cpu")
        self.assertEqual(valid.sum(1).tolist(), [7, 4])
        self.assertEqual(ids[0, -1].item(), tokenizer.eos_token_id)
        self.assertEqual(tokenizer.decode(ids[1]), "肺")
        self.assertEqual(tokenizer.decode([1, 100, 2, 101]), "a")
        self.assertTrue(ids[1, 4:].eq(tokenizer.pad_token_id).all())

    def test_pixel_preparation_preserves_aspect_and_uint8_equivalence(self):
        decoder = TaskDecoder(tiny_config()).eval()
        pixels = decoder.prepare_images([Image.new("RGB", (4, 8), color="white")])
        self.assertEqual(pixels.shape, (1, 3, 16, 16))
        self.assertEqual(pixels.device.type, "cpu")
        self.assertTrue(pixels[:, :, :, :4].eq(0).all())
        self.assertTrue(pixels[:, :, :, 4:12].eq(1).all())
        uint8 = (pixels * 255).to(torch.uint8)
        torch.testing.assert_close(decoder(prepared=pixels), decoder(prepared=uint8))
        torch.testing.assert_close(decoder(images=[Image.new("RGB", (4, 8), color="white")]),
                                   decoder(prepared=pixels))

    def test_arm_initialization_is_identical_and_baseline_uses_only_raw_path(self):
        torch.manual_seed(3)
        baseline = TaskDecoder(tiny_config(slot_conditioning=False))
        torch.manual_seed(3)
        slots_arm = TaskDecoder(tiny_config(slot_conditioning=True))
        self.assertEqual(set(baseline.state_dict()), set(slots_arm.state_dict()))
        for key, value in baseline.state_dict().items():
            torch.testing.assert_close(value, slots_arm.state_dict()[key])
        pixels = torch.rand(1, 3, 16, 16, requires_grad=True)
        with patch.object(baseline.slot_projection, "forward", side_effect=AssertionError("Slots called")):
            output = baseline(images=pixels)
            (output * torch.randn_like(output)).sum().backward()
        self.assertGreater(pixels.grad.abs().sum().item(), 0)
        self.assertGreater(baseline.patch_projection.weight.grad.abs().sum().item(), 0)
        self.assertIsNone(baseline.slot_projection[1].weight.grad)
        with self.assertRaisesRegex(ValueError, "requires images, reports"):
            baseline(slots=torch.randn(1, 8, 1024))

    def test_joint_image_report_and_all_eight_slots_reach_vqa_loss(self):
        trunk = TaskDecoder(tiny_config())
        decoder = TextDecoder(tiny_config())
        pixels = torch.rand(2, 3, 16, 16, requires_grad=True)
        slots = torch.randn(2, 8, 1024, requires_grad=True)
        features = trunk(images=pixels, reports=["edema", "clear"], slots=slots)
        self.assertEqual(features.shape, (2, 4, 16))
        loss = decoder.loss(features, ["Edema?", "Normal?"], ['["yes"]', '["no"]'])
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(pixels.grad.abs().sum().item(), 0)
        self.assertTrue(slots.grad.abs().sum((0, 2)).gt(0).all())
        self.assertGreater(trunk.report_embedding.weight.grad.abs().sum().item(), 0)
        self.assertGreater(trunk.patch_projection.weight.grad.abs().sum().item(), 0)
        self.assertGreater(decoder.embedding.weight.grad.abs().sum().item(), 0)
        self.assertGreater(decoder.output_head.weight.grad.abs().sum().item(), 0)

    def test_report_only_readout_and_padded_report_batch_are_invariant(self):
        trunk = TaskDecoder(tiny_config()).eval()
        single = trunk(reports=["edema"])
        batched = trunk(reports=["edema", "A much longer report with no effusion."])
        self.assertEqual(single.shape, (1, 1, 16))
        torch.testing.assert_close(single, batched[:1], rtol=1e-5, atol=1e-6)
        self.assertFalse(torch.allclose(single, trunk(reports=["clear"])))
        pixels = torch.rand(1, 3, 16, 16)
        mixed = trunk(images=pixels, reports=["edema"])
        padded = trunk(images=pixels.expand(2, -1, -1, -1),
                       reports=["edema", "A much longer report with no effusion."])
        torch.testing.assert_close(mixed, padded[:1], rtol=1e-5, atol=1e-6)
        self.assertFalse(torch.allclose(mixed, trunk(images=pixels)))

    def test_answer_logits_cannot_observe_future_tokens(self):
        decoder = TextDecoder(tiny_config()).eval()
        features = torch.randn(1, 4, 16)
        before = torch.tensor([[1, 10, 11, 12, 13]])
        after = torch.tensor([[1, 10, 11, 50, 51]])
        first = decoder.logits(features, ["Edema?"], before)
        second = decoder.logits(features, ["Edema?"], after)
        torch.testing.assert_close(first[:, :3], second[:, :3], rtol=0, atol=0)
        self.assertFalse(torch.allclose(first[:, 3:], second[:, 3:]))
        # Appending targets also cannot change predictions for an earlier prefix.
        prefix = decoder.logits(features, ["Edema?"], before[:, :3])
        torch.testing.assert_close(first[:, :3], prefix, rtol=1e-5, atol=1e-6)

    def test_question_padding_and_answer_padding_do_not_change_valid_logits(self):
        decoder = TextDecoder(tiny_config()).eval()
        features = torch.randn(1, 4, 16)
        single = decoder.logits(features, ["Q?"], torch.tensor([[1, 10, 11]]))
        ids = torch.tensor([[1, 10, 11, 0, 0], [1, 10, 11, 12, 13]])
        together = decoder.logits(features.expand(2, -1, -1), ["Q?", "A much longer question?"], ids)
        torch.testing.assert_close(single, together[:1, :3], rtol=1e-5, atol=1e-6)
        other = decoder.logits(features, ["Different question?"], torch.tensor([[1, 10, 11]]))
        self.assertFalse(torch.allclose(single, other))

    def test_generation_preserves_serialization_and_stops_each_row_at_eos(self):
        decoder = TextDecoder(tiny_config()).eval()
        expected = ["[]", '["yes"]']
        scripts = [decoder.tokenizer.encode(answer) + [2] for answer in expected]
        calls = []

        def scripted(memory, mask, ids):
            calls.append(ids.clone())
            output = torch.full((2, ids.shape[1], 259), -100.)
            index = ids.shape[1] - 1
            for row, script in enumerate(scripts):
                output[row, -1, script[index] if index < len(script) else 88] = 100
            return output

        with patch.object(decoder, "decode_logits", side_effect=scripted):
            actual = decoder.generate(torch.randn(2, 4, 16), ["Q1?", "Q2?"])
        self.assertEqual(actual, expected)
        self.assertEqual(len(calls), len(scripts[1]))
        self.assertTrue(calls[-1][0, len(scripts[0]) + 1:].eq(0).all())
        self.assertEqual(json.loads(actual[1]), ["yes"])
        with self.assertRaises(ValueError):
            decoder.train().generate(torch.randn(1, 4, 16), ["Q?"], 3)


if __name__ == "__main__":
    unittest.main()
