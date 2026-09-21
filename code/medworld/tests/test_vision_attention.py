import copy
from unittest.mock import patch
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionAttention, Qwen3_5VisionModel
from medworld.vision_attention import BatchedVisionAttention, batch_vision_attention


def compare(lengths):
    torch.manual_seed(17)
    config = Qwen3_5VisionConfig(hidden_size=32, num_heads=4)
    config._attn_implementation = "sdpa"
    original = Qwen3_5VisionAttention(config).double()
    batched = copy.deepcopy(original)
    batched.__class__ = BatchedVisionAttention
    total = sum(lengths)
    inputs = torch.randn(total, 32, dtype=torch.float64, requires_grad=True)
    other = inputs.detach().clone().requires_grad_()
    positions = torch.randn(total, 8, dtype=torch.float64)
    embeddings = (positions.cos(), positions.sin())
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    a, b = original(inputs, cu, embeddings), batched(other, cu, embeddings)
    torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)
    a.square().sum().backward()
    b.square().sum().backward()
    torch.testing.assert_close(inputs.grad, other.grad, rtol=1e-10, atol=1e-10)
    assert dict(original.named_parameters()).keys() == dict(batched.named_parameters()).keys()
    for left, right in zip(original.parameters(), batched.parameters()):
        torch.testing.assert_close(left.grad, right.grad, rtol=1e-10, atol=1e-10)
    return batched, inputs, cu, embeddings, b


def test_batched_vision_outputs_gradients_and_image_isolation():
    batched, inputs, cu, embeddings, output = compare([12, 12, 12])
    changed = inputs.detach().clone()
    changed[12:] += 100
    torch.testing.assert_close(batched(changed, cu, embeddings)[:12], output[:12])


def test_variable_length_fallback():
    compare([8, 12, 6])


def test_full_vision_model_uses_one_attention_call_per_layer():
    config = Qwen3_5VisionConfig(hidden_size=32, num_heads=4, depth=1,
        intermediate_size=64, out_hidden_size=32, patch_size=16,
        temporal_patch_size=2, spatial_merge_size=2, num_position_embeddings=64)
    config._attn_implementation = "sdpa"
    model = Qwen3_5VisionModel(config).eval()
    batch_vision_attention(model)
    original = torch.nn.functional.scaled_dot_product_attention
    with patch("torch.nn.functional.scaled_dot_product_attention", wraps=original) as attention:
        model(hidden_states=torch.randn(32, 1536), grid_thw=torch.tensor([[1, 4, 4], [1, 4, 4]]))
    assert attention.call_count == 1
