"""Real-model gradient and reload checks for image-conditioned tasks."""
import torch
from .downstream_tasks.registry import TASKS


def audit_model(model, data, out):
    model.eval()
    conditioned = model.cfg['slot_conditioning']
    result = {'tasks': {}, 'slot_conditioning': conditioned,
              'architecture': model.cfg['architecture']}
    if not conditioned:
        forbidden = ('encoder', 'jepa', 'world', 'visual_teacher', 'visual_reconstruction')
        present = [name for name in forbidden if hasattr(model, name)]
        if present or model.target is not None:
            raise AssertionError(f'Raw baseline contains slot-branch components: {present}, target={model.target}')
        result['baseline_slot_branch_absent'] = True
    for task in TASKS:
        model.zero_grad(set_to_none=True)
        batch = data.batch(task, 'validate', [0])
        loss, _ = model(task, batch)
        loss.backward()
        image_module = model.task_decoder.patch_projection
        vision = sum(float(p.grad.float().norm()) for p in image_module.parameters() if p.grad is not None)
        slot_grad = model.encoder.slot_queries.grad if conditioned else None
        slots = float(slot_grad.norm()) if slot_grad is not None else 0.
        if vision <= 0 or (conditioned and slots <= 0):
            raise AssertionError(f'{task}: missing image/slot gradient')
        result['tasks'][task] = {'loss': float(loss.detach()), 'raw_patch_gradient': vision, 'slot_gradient': slots}
    model.zero_grad(set_to_none=True)
    from .runtime import load_model
    restored, _ = load_model(out / 'final.pt', str(model.device))
    batch = data.batch('classification', 'validate', [0])
    before, after = model.predict('classification', batch), restored.predict('classification', batch)
    torch.testing.assert_close(before, after, atol=1e-5, rtol=1e-5)
    result['checkpoint_reload_equal'] = True
    return result
