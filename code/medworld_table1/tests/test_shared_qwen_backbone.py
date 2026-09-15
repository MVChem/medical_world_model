"""CPU contracts for shared frozen bases and Qwen's untied language head."""
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import model as model_module
from model import MedWorld, copy_with_shared_frozen_parameters
from ablation_model import AblationMedWorld, Memory
from medworld_common.qwen import ReportDecoder, hidden


class TinyTokenizer:
    eos_token_id = 6

    def apply_chat_template(self, *args, **kwargs):
        return [1]

    def batch_decode(self, ids, **kwargs):
        return [' '.join(map(str, row.tolist())) for row in ids]


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(7, 4)
        self.linear = nn.Linear(4, 4, bias=False)
        self.register_buffer('cached_scale', torch.tensor(1.0))

    def get_input_embeddings(self):
        return self.embed_tokens

    def gradient_checkpointing_enable(self, **kwargs):
        self.checkpointing = True

    def gradient_checkpointing_disable(self):
        self.checkpointing = False

    def forward(self, inputs_embeds, attention_mask, **kwargs):
        # Causal mixing makes report loss depend on the clinical prefix.
        values = torch.cumsum(inputs_embeds * attention_mask[:, -inputs_embeds.shape[1]:, None], 1)
        values = self.linear(values)
        if hasattr(self, 'lora_A'):
            values = values + F.linear(values, self.lora_A)
        return SimpleNamespace(last_hidden_state=values, past_key_values=None)


def fake_model(monkeypatch, *, shared, untied=False, condition='slots'):
    def load(_):
        backbone = TinyBackbone()
        base = nn.Module()
        base.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=4), tie_word_embeddings=not untied)
        base.model = nn.Module()
        base.model.language_model = backbone
        base.lm_head = nn.Linear(4, 7, bias=False)
        return base.requires_grad_(False)

    def adapt(backbone, _):
        backbone.lora_A = nn.Parameter(torch.randn(4, 4) * .02)
        return backbone

    monkeypatch.setattr(model_module, 'load_qwen', load)
    monkeypatch.setattr(model_module, 'adapt', adapt)
    monkeypatch.setattr(model_module.AutoTokenizer, 'from_pretrained', lambda *a, **k: TinyTokenizer())
    cfg = dict(qwen='unused', lwm_width=16, lwm_depth=1, slots=2, visual_grid=1,
               findings=['a', 'b', 'c'], latent_weight=1., text_weight=1., finding_weight=.5,
               replay_weight=.1, share_frozen_backbone=shared, state_condition=condition, generation_tokens=1)
    torch.manual_seed(19)
    return AblationMedWorld(cfg)


def batch():
    torch.manual_seed(3)
    values = {'horizon': torch.tensor([0, 1])}
    for side in ('source', 'target'):
        values[side+'_features'] = torch.randn(2, 1, 768)
        values[side+'_ids'] = torch.tensor([[1, 2], [2, 3]])
        values[side+'_mask'] = torch.ones(2, 2, dtype=torch.long)
        values[side+'_target_ids'] = torch.tensor([[3, 4], [4, 5]])
        values[side+'_target_mask'] = torch.ones(2, 2, dtype=torch.long)
        values[side+'_labels'] = torch.tensor([[1, 0, 1], [0, 1, 0]])
    return values


@pytest.mark.parametrize('condition', ['slots', 'no_slots'])
def test_shared_base_matches_independent_loss_gradients_and_keeps_target_fixed(monkeypatch, condition):
    independent = fake_model(monkeypatch, shared=False, untied=True, condition=condition)
    shared = fake_model(monkeypatch, shared=True, untied=True, condition=condition)
    independent.begin_stage2()
    shared.begin_stage2()
    for current in (independent, shared):
        current.train()
    inputs = batch()
    expected, _ = independent.losses(inputs, 2)
    actual, _ = shared.losses(inputs, 2)
    torch.testing.assert_close(actual, expected)
    expected.backward()
    actual.backward()
    expected_parameters = dict(independent.named_parameters())
    for name, parameter in shared.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            torch.testing.assert_close(parameter.grad, expected_parameters[name].grad)
    assert shared.encoder.backbone.linear.weight is shared.decoder.backbone.linear.weight
    assert shared.encoder.backbone.linear.weight is shared.target_encoder.backbone.linear.weight
    assert shared.encoder.backbone.lora_A is not shared.decoder.backbone.lora_A
    assert shared.encoder.backbone.lora_A is not shared.target_encoder.backbone.lora_A
    assert shared.encoder.backbone.lora_A.requires_grad
    assert all(not p.requires_grad and p.grad is None for p in shared.target_encoder.parameters())
    assert shared.encoder.backbone.checkpointing
    assert not shared.target_encoder.backbone.checkpointing
    old_target = {n: p.detach().clone() for n, p in shared.target_encoder.named_parameters()}
    old_lora = shared.encoder.backbone.lora_A.detach().clone()
    optimizer = torch.optim.SGD([p for p in shared.parameters() if p.requires_grad], lr=.05)
    optimizer.step()
    assert not torch.equal(old_lora, shared.encoder.backbone.lora_A)
    for name, parameter in shared.target_encoder.named_parameters():
        torch.testing.assert_close(parameter, old_target[name], rtol=0, atol=0)
    shared.to(dtype=torch.float64)
    assert shared.encoder.backbone.linear.weight is shared.decoder.backbone.linear.weight
    assert shared.encoder.backbone.linear.weight is shared.target_encoder.backbone.linear.weight
    assert shared.encoder.backbone.linear.weight.dtype == torch.float64
    audit = shared.audit_shared_backbones()
    assert audit['enabled'] and audit['target_frozen'] and audit['online_lora_trainable']
    assert audit['devices'] == ['cpu']


def test_buffers_and_trainable_parameters_are_independent():
    source = TinyBackbone().requires_grad_(False)
    source.lora_A = nn.Parameter(torch.randn(4, 4))
    cloned = copy_with_shared_frozen_parameters(source)
    assert cloned.embed_tokens.weight is source.embed_tokens.weight
    assert cloned.lora_A is not source.lora_A
    cloned.cached_scale.fill_(3.)
    assert source.cached_scale.item() == 1.
    cloned.requires_grad_(False)
    assert source.lora_A.requires_grad


def test_real_peft_keeps_shared_qwen_base_and_independent_adapters():
    from transformers import Qwen3_5TextConfig, Qwen3_5TextModel
    from medworld_common.qwen import adapt
    from model import audit_frozen_parameter_copy
    config = Qwen3_5TextConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                              num_hidden_layers=1, num_attention_heads=2,
                              num_key_value_heads=1, head_dim=8, layer_types=['full_attention'])
    source = Qwen3_5TextModel(config).requires_grad_(False)
    decoder = copy_with_shared_frozen_parameters(source)
    source = adapt(source, dict(lora_rank=2, lora_alpha=4))
    decoder = adapt(decoder, dict(lora_rank=2, lora_alpha=4))
    audit = audit_frozen_parameter_copy(source, decoder)
    assert audit['shared_frozen_parameters'] > 0
    assert audit['independent_trainable_parameters'] > 0
    target = copy_with_shared_frozen_parameters(source).requires_grad_(False)
    assert all(not p.requires_grad for p in target.parameters())
    assert all(p.requires_grad for name, p in source.named_parameters() if 'lora_' in name)


@pytest.mark.parametrize('full_memory', [False, True])
def test_report_loss_and_generation_use_untied_output_head(full_memory):
    torch.manual_seed(5)
    backbone = TinyBackbone().requires_grad_(False)
    output_head = nn.Linear(4, 7, bias=False)
    decoder = ReportDecoder(backbone, TinyTokenizer(), 4, output_head=output_head)
    assert decoder.language_weight() is output_head.weight
    assert not output_head.weight.requires_grad
    state = torch.randn(2, 2, 4, requires_grad=True)
    ids, mask = torch.tensor([[3, 4], [4, 5]]), torch.ones(2, 2, dtype=torch.long)
    ablation = AblationMedWorld.__new__(AblationMedWorld)
    nn.Module.__init__(ablation)
    ablation.decoder = decoder
    ablation.tokenizer = decoder.tokenizer
    ablation.cfg = dict(generation_tokens=1)
    if full_memory:
        memory = Memory(state, torch.tensor([[1, 0], [1, 1]]))
        prefix, prefix_mask = ablation.prefix(memory)
        actual = ablation.report_loss(memory, ids, mask)
    else:
        memory = state
        prefix = decoder.prefix(state)
        prefix_mask = torch.ones(prefix.shape[:2], dtype=torch.long)
        actual = decoder.loss(state, ids, mask)
    outputs = hidden(backbone, torch.cat([prefix, backbone.embed_tokens(ids)], 1),
                     torch.cat([prefix_mask, mask], 1)).last_hidden_state
    logits = F.linear(outputs[:, prefix.shape[1]-1:-1], output_head.weight)
    expected = F.cross_entropy(logits.reshape(-1, 7), ids.flatten())
    wrong = F.cross_entropy(F.linear(outputs[:, prefix.shape[1]-1:-1], backbone.embed_tokens.weight).reshape(-1, 7), ids.flatten())
    torch.testing.assert_close(actual, expected)
    assert not torch.isclose(actual, wrong)
    actual.backward()
    assert state.grad is not None and state.grad.abs().sum() > 0
    assert output_head.weight.grad is None
    first = hidden(backbone, prefix, prefix_mask).last_hidden_state[:, -1]
    expected_reports = decoder.tokenizer.batch_decode(F.linear(first, output_head.weight).argmax(-1)[:, None])
    if full_memory:
        ablation.state = lambda batch: memory
        ablation.forecast = lambda state, horizon: state
        ablation.scores = lambda state: torch.zeros(2, 1)
        reports, _ = ablation.predict({'horizon': torch.zeros(2, dtype=torch.long)})
    else:
        reports = decoder.generate(state, 1)
    assert reports == expected_reports


@pytest.mark.parametrize('untied', [False, True])
def test_compact_checkpoint_keeps_adapters_and_omits_pretrained_output_head(monkeypatch, untied):
    current = fake_model(monkeypatch, shared=True, untied=untied)
    current.begin_stage2()
    compact = copy.deepcopy(current.compact_state())
    assert not any(key.startswith('decoder.output_head.') for key in compact)
    assert 'encoder.backbone.lora_A' in compact
    assert 'target_encoder.backbone.lora_A' in compact
    current.load_compact(compact)
    assert (current.decoder.output_head is not None) == untied
    if not untied:
        assert current.decoder.language_weight() is current.decoder.backbone.embed_tokens.weight
