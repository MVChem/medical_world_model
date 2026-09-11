"""GPU integration checks using the real small Qwen checkpoint."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import pytest
import torch
from model import MedWorld, chunked_ce


def test_chunked_ce_matches_dense_value_and_gradients():
    torch.manual_seed(1)
    weight = torch.randn(19, 7)
    states = torch.randn(2, 5, 7, requires_grad=True)
    targets = torch.randint(0, 19, (2, 5))
    targets[0, 3:] = -100
    loss = chunked_ce(states, targets, weight, chunk=3)
    loss.backward()
    gradient = states.grad.clone()
    states.grad = None
    dense = torch.nn.functional.cross_entropy(torch.nn.functional.linear(states, weight).flatten(0, 1), targets.flatten())
    dense.backward()
    torch.testing.assert_close(loss, dense)
    torch.testing.assert_close(gradient, states.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU integration test')
def test_future_is_target_only_and_target_parameters_stay_frozen():
    cfg = common.read_config(Path(__file__).resolve().parents[1] / 'configs/pilot.json')
    model = MedWorld(cfg).cuda().eval()
    model.begin_stage2()
    ids = torch.tensor([[100, 200, 300]], device='cuda')
    batch = {'horizon': torch.tensor([0], device='cuda')}
    for side in ('source', 'target'):
        batch[side+'_ids'] = ids.clone()
        batch[side+'_mask'] = torch.ones_like(ids)
        batch[side+'_features'] = torch.randn(1, 64, 768, device='cuda')
        batch[side+'_target_ids'] = ids.clone()
        batch[side+'_target_mask'] = torch.ones_like(ids)
        batch[side+'_labels'] = torch.zeros(1, 6, device='cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        prediction = model.world(model.state(batch), batch['horizon'])
        batch['target_ids'].fill_(999)
        batch['target_features'].mul_(8)
        altered = model.world(model.state(batch), batch['horizon'])
        torch.testing.assert_close(prediction, altered, rtol=0, atol=0)
        loss, _ = model.losses(batch, 2)
    loss.backward()
    assert model.encoder.slots.grad is not None
    assert all(p.grad is None for p in model.target_encoder.parameters())
    assert any(p.grad is not None for p in model.encoder.adapter.parameters())
    assert any(p.grad is not None for name, p in model.encoder.named_parameters() if 'lora_' in name)
    assert any(p.grad is not None for p in model.world.parameters())
