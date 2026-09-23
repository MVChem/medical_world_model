import pytest
import torch

from medworld.downstream_tasks.future.generation import ByteDecoderCache, generate_cached
from medworld.downstream_tasks.text import TextDecoder


@pytest.mark.parametrize("dtype,tolerance", [(torch.float32, 2e-6), (torch.bfloat16, .03)])
def test_incremental_logits_match_full_causal_decoder_with_padding(dtype, tolerance):
    torch.manual_seed(37)
    decoder = TextDecoder({"decoder_width": 16, "decoder_depth": 2, "context_tokens": 32,
                           "answer_tokens": 32, "generation_tokens": 32}).to(dtype).eval()
    features = torch.randn(2, 4, 16).to(dtype)
    questions = ["Report?", "A longer question"]
    # Padding begins after EOS in row 0; row 1 continues generating.
    ids = torch.tensor([[1, 40, 80, 2, 0, 0], [1, 35, 90, 100, 85, 2]])
    with torch.no_grad():
        cache = ByteDecoderCache(decoder, features, questions, ids.shape[1])
        for i in range(ids.shape[1]):
            actual = cache.step(ids[:, i:i + 1])[:, -1]
            expected = decoder.logits(features, questions, ids[:, :i + 1])[:, -1]
            torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


def test_incremental_greedy_outputs_match_and_preserve_parameters():
    torch.manual_seed(47)
    decoder = TextDecoder({"decoder_width": 16, "decoder_depth": 2, "context_tokens": 32,
                           "answer_tokens": 32, "generation_tokens": 32}).eval()
    features = torch.randn(3, 4, 16)
    questions = ["Report?", "A longer question", "Another"]
    saved = {name: value.clone() for name, value in decoder.state_dict().items()}
    expected = decoder.generate(features, questions, 32)
    assert generate_cached(decoder, features, questions, 32) == expected
    for name, value in decoder.state_dict().items():
        torch.testing.assert_close(value, saved[name], rtol=0, atol=0)
    with torch.no_grad():
        decoder.output_head.weight.zero_()
        decoder.output_head.bias.zero_()
        decoder.output_head.bias[decoder.tokenizer.eos_token_id] = 10
    assert generate_cached(decoder, features, questions, 32) == ["", "", ""]


def test_cache_rejects_training_and_out_of_bounds_steps():
    decoder = TextDecoder({"decoder_width": 16, "decoder_depth": 1, "context_tokens": 8,
                           "answer_tokens": 8, "generation_tokens": 8})
    features = torch.randn(1, 2, 16)
    with pytest.raises(ValueError, match="eval"):
        ByteDecoderCache(decoder, features, ["Report"], 1)
    decoder.eval()
    with torch.no_grad():
        cache = ByteDecoderCache(decoder, features, ["Report"], 1)
        cache.step(torch.tensor([[1]]))
        with pytest.raises(ValueError, match="budget"):
            cache.step(torch.tensor([[20]]))
