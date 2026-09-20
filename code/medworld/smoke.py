"""Real-model gradient and reload checks for image-conditioned tasks."""
import torch
from .downstream_tasks.registry import TASKS


def audit_model(model, data, out):
    model.eval()
    result = {'tasks': {}, 'slot_conditioning': model.cfg['slot_conditioning']}
    for task in TASKS:
        model.zero_grad(set_to_none=True)
        batch = data.batch(task, 'validate', [0])
        loss, _ = model(task, batch)
        loss.backward()
        vision = sum(float(p.grad.float().norm()) for p in model.encoder.vision.parameters() if p.grad is not None)
        slot_grad = model.encoder.slot_queries.grad
        slots = float(slot_grad.norm()) if slot_grad is not None else 0.
        if vision <= 0 or (model.cfg['slot_conditioning'] and slots <= 0):
            raise AssertionError(f'{task}: missing image/slot gradient')
        result['tasks'][task] = {'loss': float(loss.detach()), 'vision_gradient': vision, 'slot_gradient': slots}
    model.zero_grad(set_to_none=True)
    from .runtime import load_model
    restored, _ = load_model(out / 'final.pt', str(model.device))
    batch = data.batch('classification', 'validate', [0])
    before, after = model.predict('classification', batch), restored.predict('classification', batch)
    torch.testing.assert_close(before, after, atol=1e-5, rtol=1e-5)
    result['checkpoint_reload_equal'] = True
    return result
