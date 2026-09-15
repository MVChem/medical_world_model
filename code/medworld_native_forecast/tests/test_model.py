"""CPU checks for immutable target sharing, native image positions, and reload."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model


spec = importlib.util.spec_from_file_location(
    "native_forecast_model_test_module", Path(__file__).resolve().parents[1] / "model.py")
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


class SmallBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = model.LoRALinear(nn.Linear(4, 4).requires_grad_(False), 2, 2)
        self.checkpointing = True

    def gradient_checkpointing_disable(self):
        self.checkpointing = False

    def forward(self, inputs):
        return self.projection(inputs)


class SmallEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.language, self.vision = SmallBranch(), SmallBranch()
        self.readout = nn.Linear(4, 4)
        self.register_buffer("buffer", torch.randn(4))

    def forward(self, inputs):
        return self.readout(self.language(inputs) + self.vision(inputs)) + self.buffer


def compact_model(native=False):
    instance = model.NativeForecast.__new__(model.NativeForecast)
    nn.Module.__init__(instance)
    instance.decoder = nn.Module()
    instance.decoder.language = SmallBranch()
    instance.finding = nn.Linear(4, 2)
    instance.encoder = None if native else SmallEncoder()
    instance.target_encoder = None
    return instance


def test_target_copy_shares_only_immutable_parameters_and_stays_fixed():
    torch.manual_seed(7)
    online = SmallEncoder()
    target = model.shared_frozen_copy(online).requires_grad_(False).eval()
    assert target.language.projection.base.weight is online.language.projection.base.weight
    assert target.language.projection.lora_a is not online.language.projection.lora_a
    assert target.language.projection.lora_b is not online.language.projection.lora_b
    assert target.readout.weight is not online.readout.weight
    assert target.buffer is not online.buffer
    inputs = torch.randn(3, 4)
    before = target(inputs).detach().clone()
    optimizer = torch.optim.SGD([p for p in online.parameters() if p.requires_grad], lr=0.3)
    online(inputs).square().mean().backward()
    assert online.language.projection.lora_b.grad.norm() > 0
    assert all(parameter.grad is None for parameter in target.parameters())
    optimizer.step()
    torch.testing.assert_close(target(inputs), before, rtol=0, atol=0)
    assert not torch.equal(online(inputs), before)


def test_stage1_transfer_and_stage2_compact_restore_preserve_fixed_target():
    torch.manual_seed(2)
    original = compact_model()
    stage1 = original.compact_state()
    initialized = compact_model()
    initialized.load_compact(stage1)
    initialized.begin_stage2()
    initialized.train()
    assert not initialized.target_encoder.training
    assert not initialized.target_encoder.language.checkpointing
    assert not initialized.target_encoder.vision.checkpointing
    inputs = torch.randn(2, 4)
    torch.testing.assert_close(initialized.encoder(inputs), initialized.target_encoder(inputs))
    with torch.no_grad():
        initialized.encoder.readout.weight.add_(1)
    stage2 = initialized.compact_state()
    assert not any(".base." in key for key in stage2)
    assert any(key.startswith("target_encoder.") for key in stage2)
    restored = compact_model()
    # Compact checkpoints intentionally reference original frozen pretrained bases.
    for name, parameter in restored.named_parameters():
        if not parameter.requires_grad:
            parameter.data.copy_(dict(initialized.named_parameters())[name])
    restored.load_compact(stage2)
    assert restored.compact_state().keys() == stage2.keys()
    assert all(torch.equal(value, stage2[key]) for key, value in restored.compact_state().items())
    torch.testing.assert_close(restored.target_encoder(inputs), initialized.target_encoder(inputs))
    assert not torch.equal(restored.encoder(inputs), restored.target_encoder(inputs))
    native = compact_model(native=True)
    native.load_compact(stage1)
    assert all(torch.equal(value, stage1[key]) for key, value in native.compact_state().items())


def test_partial_stage2_target_checkpoint_is_rejected():
    instance = compact_model()
    instance.begin_stage2()
    state = instance.compact_state()
    del state["target_encoder.readout.weight"]
    with pytest.raises(ValueError, match="target_encoder.readout.weight"):
        compact_model().load_compact(state)


class RecordingLanguage(nn.Module):
    def __init__(self):
        super().__init__()
        self.positions = []

    def forward(self, inputs_embeds=None, input_ids=None, position_ids=None, **kwargs):
        self.positions.append(position_ids.detach().clone())
        shape = inputs_embeds.shape[:2] if inputs_embeds is not None else input_ids.shape
        return SimpleNamespace(last_hidden_state=torch.zeros(*shape, 4), past_key_values=object())


class TinyNative(nn.Module):
    get_rope_index = Qwen3_5Model.get_rope_index
    get_vision_position_ids = Qwen3_5Model.get_vision_position_ids

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(64, 4)
        self.language_model = RecordingLanguage()
        self.config = SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2))

    def get_input_embeddings(self):
        return self.embedding

    def get_image_features(self, pixels, grid, return_dict=True):
        return SimpleNamespace(pooler_output=tuple(torch.full((4, 4), float(i + 1)) for i in range(len(grid))))

    def get_placeholder_mask(self, ids, inputs_embeds=None, image_features=None):
        return (ids == 7).unsqueeze(-1).expand_as(inputs_embeds), None


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 3

    def convert_tokens_to_ids(self, token):
        assert token == "<|im_end|>"
        return 20

    def batch_decode(self, ids, skip_special_tokens=True):
        return [str(row.tolist()) for row in ids]


def native_model_and_inputs():
    instance = compact_model(native=True)
    instance.decoder.model = TinyNative()
    instance.decoder.lm_head = nn.Linear(4, 64, bias=False)
    instance.tokenizer = TinyTokenizer()
    instance.slot_projection = nn.Linear(model.WIDTH, 4)
    instance.cfg = dict(generation_tokens=2)
    ids = torch.tensor([[0, 0, 11, 7, 7, 7, 7, 12, 13, 20, 21, 22],
                        [10, 11, 11, 7, 7, 7, 7, 12, 13, 20, 21, 22]])
    inputs = dict(input_ids=ids, attention_mask=(ids != 0).long(),
                  mm_token_type_ids=(ids == 7).long(), image_grid_thw=torch.tensor([[1, 4, 4]] * 2),
                  pixel_values=torch.zeros(8, 4))
    return instance, inputs


def test_slot_prefix_preserves_native_image_embeddings_and_multimodal_coordinates():
    instance, inputs = native_model_and_inputs()
    native = instance.native_prefix(inputs)
    slots = instance.native_prefix(inputs, torch.randn(2, 8, model.WIDTH))
    cut, count = 9, 8
    torch.testing.assert_close(slots["embeds"][:, :cut], native["embeds"][:, :cut])
    torch.testing.assert_close(slots["embeds"][:, cut + count:], native["embeds"][:, cut:])
    torch.testing.assert_close(slots["positions"][:, :, :cut], native["positions"][:, :, :cut])
    torch.testing.assert_close(slots["positions"][:, :, cut + count:], native["positions"][:, :, cut:] + count)
    assert slots["mask"][:, cut:cut + count].eq(1).all()
    assert slots["embeds"][0, 3:7].eq(1).all()
    assert slots["embeds"][1, 3:7].eq(2).all()
    torch.testing.assert_close(slots["delta"], native["delta"])


def test_cached_generation_position_accounts_for_left_padding():
    instance, inputs = native_model_and_inputs()
    prefix = instance.native_prefix(inputs)
    instance.source_inputs = lambda batch: inputs
    instance.predict({})
    actual = instance.decoder.model.language_model.positions[1]
    expected = prefix["positions"].amax(dim=(0, 2)) + 1
    torch.testing.assert_close(actual[:, :, 0], expected[None].expand(3, -1))
