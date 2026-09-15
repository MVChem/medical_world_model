"""Contracts for full-token masking and patient-disjoint source-state donors."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ablation_model import AblationMedWorld, Memory, pooled_memory
from data import Corpus
from model import WorldModel


def test_full_token_prediction_ignores_padding_and_retains_valid_gradients():
    torch.manual_seed(3)
    model = AblationMedWorld.__new__(AblationMedWorld)
    torch.nn.Module.__init__(model)
    model.world = WorldModel(dict(lwm_width=16, lwm_depth=1, slots=8), 16)
    # WorldModel starts with zero residual; exercise the actual attention path.
    torch.nn.init.normal_(model.world.output.weight, std=.05)
    values = torch.randn(2, 11, 16, requires_grad=True)
    mask = torch.ones(2, 11, dtype=torch.long)
    mask[0, 4:7] = 0
    horizon = torch.tensor([0, 1])
    pred = model.forecast(Memory(values, mask), horizon)
    altered = values.detach().clone()
    altered[0, 4:7] = 999
    other = model.forecast(Memory(altered, mask), horizon)
    torch.testing.assert_close(pred.values[mask.bool()], other.values[mask.bool()])
    longer_values = torch.cat([values.detach(), torch.randn(2, 4, 16)], 1)
    longer_mask = torch.cat([mask, torch.zeros(2, 4, dtype=torch.long)], 1)
    longer = model.forecast(Memory(longer_values, longer_mask), horizon)
    torch.testing.assert_close(pred.values[mask.bool()], longer.values[longer_mask.bool()])
    pooled_memory(pred).square().mean().backward()
    assert torch.isfinite(values.grad).all()
    assert values.grad[mask.bool()].abs().sum() > 0
    assert values.grad[~mask.bool()].abs().sum() == 0
    assert pred.values.shape == values.shape


def test_state_donors_are_deterministic_other_patients_and_same_split(tmp_path):
    observations = []
    pairs = {}
    for split in ('train', 'validate', 'test'):
        pairs[split] = []
        for i in range(3):
            id = f'{split}-{i}'
            observations.append(dict(id=id, patient=id, split=split, report=id, image=id, labels=[0]))
            pairs[split].append(dict(id=id, patient=id, source=id, target=id, horizon=0))
        (tmp_path / f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in pairs[split]))
    (tmp_path / 'observations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in observations))
    np.save(tmp_path / 'vjepa_features.npy', np.arange(9, dtype=np.float16)[:, None, None])

    class Tokenizer:
        eos_token_id = 2
        pad_token_id = 0

        def __call__(self, texts, **kwargs):
            return {'input_ids': [[3] for _ in texts]}

    cfg = dict(cache=str(tmp_path), seed=42, report_tokens=8, state_condition='shuffled')
    first, second = Corpus(cfg, Tokenizer()), Corpus(cfg, Tokenizer())
    assert first.donors == second.donors
    for split, rows in pairs.items():
        for row in rows:
            assert first.donors[row['id']] != row['source']
            assert first.donors[row['id']].startswith(split)
        batch = first.batch(rows[:1], device='cpu')
        assert batch['donor_features'].item() != batch['source_features'].item()
