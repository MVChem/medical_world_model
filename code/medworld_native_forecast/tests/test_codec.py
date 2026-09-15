"""State-only report API: no observations, mixed EOS, validation, cache positions."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

spec = importlib.util.spec_from_file_location('native_state_codec', Path(__file__).resolve().parents[1] / 'codec.py')
codec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(codec)


class Language(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        # First sample ends immediately; second emits 1, 2, EOS.
        tokens = [3, [1, 2, 3][len(self.calls)-1]]
        hidden = torch.nn.functional.one_hot(torch.tensor(tokens), 4).float()[:, None]
        return SimpleNamespace(last_hidden_state=hidden, past_key_values=len(self.calls))


class Tokenizer:
    eos_token_id, pad_token_id = 3, 0

    def batch_decode(self, ids, skip_special_tokens=True):
        self.ids = ids
        return [' '.join(str(int(t)) for t in row if int(t) not in (0, 3)) for row in ids]


class SlotsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Identity()
        self.slot_projection = nn.Linear(1024, 4)
        self.tokenizer = Tokenizer()
        self.decoder = SimpleNamespace(model=SimpleNamespace(language_model=Language()),
                                       lm_head=SimpleNamespace(weight=torch.eye(4)))
        self.device = torch.device('cpu')
        self.cfg = {'generation_tokens': 8}

    def state_only_prefix(self, state):
        self.received = state.clone()
        return dict(embeds=self.slot_projection(state), mask=torch.ones(len(state), 8, dtype=torch.long),
                    positions=torch.arange(8)[None, None].expand(3, len(state), -1),
                    delta=torch.zeros(len(state), 1, dtype=torch.long))

    def state(self, *args, **kwargs):
        raise AssertionError('Decoding must not re-encode observations')

    source_inputs = state
    native_prefix = state


def test_decode_external_state_without_observations_and_stop_per_sample():
    model = SlotsModel().eval()
    state = torch.randn(2, 8, 1024, requires_grad=True)
    assert codec.decode_reports(model, state) == ['', '1 2']
    torch.testing.assert_close(model.received, state)
    assert model.tokenizer.ids.tolist() == [[3, 0, 0], [1, 2, 3]]
    calls = model.decoder.model.language_model.calls
    assert len(calls) == 3
    assert not calls[0]['inputs_embeds'].requires_grad
    for index, call in enumerate(calls[1:], 1):
        assert call['past_key_values'] == index
        assert call['position_ids'].eq(7 + index).all()


def test_generation_budget_is_respected():
    model = SlotsModel().eval()
    assert codec.decode_reports(model, torch.randn(2, 8, 1024), max_new_tokens=1) == ['', '1']
    assert len(model.decoder.model.language_model.calls) == 1


@pytest.mark.parametrize('state', [torch.zeros(1, 8, 4096), torch.zeros(1, 4, 1024),
    torch.zeros(8, 1024), torch.zeros(0, 8, 1024), torch.zeros(1, 8, 1024, dtype=torch.long),
    torch.full((1, 8, 1024), float('nan')), torch.full((1, 8, 1024), float('inf'))])
def test_reject_incompatible_or_invalid_states(state):
    with pytest.raises(ValueError):
        codec.decode_reports(SlotsModel().eval(), state)


@pytest.mark.parametrize('limit', [0, -1, 1.5, True])
def test_reject_invalid_generation_budget(limit):
    with pytest.raises(ValueError, match='positive integer'):
        codec.decode_reports(SlotsModel().eval(), torch.zeros(2, 8, 1024), max_new_tokens=limit)


def test_require_evaluation_mode_and_state_checkpoint():
    model = SlotsModel()
    with pytest.raises(ValueError, match='eval'):
        codec.decode_reports(model, torch.zeros(2, 8, 1024))
    model.eval()
    model.encoder = None
    with pytest.raises(ValueError, match='checkpoint'):
        codec.decode_reports(model, torch.zeros(2, 8, 1024))
