"""Model/data selection and gradient contracts for the shared training loop."""

import math
import torch


def resolve_variant(cfg, requested="auto"):
    inferred = (
        "noslots"
        if cfg.get("slots") == 0
        else "slot44"
        if "qa_manifest" in cfg
        else "legacy"
    )
    variant = inferred if requested == "auto" else requested
    if variant != inferred:
        raise ValueError(f"Variant {variant} conflicts with config ({inferred})")
    return variant


def build(cfg, variant, freeze_encoder=False):
    if variant == "legacy":
        from networks import FourTaskModel
        from corpus import Corpus
        from evaluation import evaluate

        model = FourTaskModel(cfg, freeze_encoder=freeze_encoder)
    else:
        from slot44_corpus import Slot44Corpus as Corpus
        from slot44_evaluation import evaluate

        if variant == "slot44":
            from slot44_networks import Slot44Model

            model = Slot44Model(cfg, freeze_encoder=freeze_encoder)
        else:
            from noslots_networks import NoSlotsModel

            model = NoSlotsModel(cfg)
    return model, Corpus, evaluate


def gradient_record(model, task, step, freeze_encoder, variant):
    if variant == "noslots":
        memory = model.last_memory
        memory_grad = memory.values.grad
        if memory_grad is None or not torch.isfinite(memory_grad).all():
            raise RuntimeError(f"{task}: invalid direct-token gradient")
        padding_grad = float(
            (memory_grad.float() * (~memory.mask.bool())[:, :, None]).norm()
        )
        if padding_grad != 0:
            raise RuntimeError(f"{task}: decoder reads padded token positions")
        memory_grad_norm = float(memory_grad.float().norm())
        encoder_grad = math.sqrt(
            sum(
                float(p.grad.float().square().sum())
                for p in model.encoder.parameters()
                if p.requires_grad and p.grad is not None
            )
        )
        if step < 4 and (
            not math.isfinite(encoder_grad)
            or encoder_grad == 0
            or memory_grad_norm == 0
        ):
            raise RuntimeError(f"{task}: no finite nonzero gradient to shared encoder")
        return dict(
            memory_tokens=memory.mask.sum(-1).cpu().tolist(),
            memory_grad_norm=memory_grad_norm,
            padding_grad_norm=padding_grad,
            encoder_grad_norm=encoder_grad,
        )
    slot_grad = model.encoder.slots.grad
    slot_norm = float(slot_grad.float().norm()) if slot_grad is not None else 0.0
    if (
        not freeze_encoder
        and step < 4
        and (not math.isfinite(slot_norm) or slot_norm == 0)
    ):
        raise RuntimeError(f"{task}: no finite nonzero gradient to slots")
    result = dict(slot_grad_norm=slot_norm)
    if variant == "legacy" or freeze_encoder:
        return result
    state_grad = model.last_state.grad
    route_norms = (
        state_grad.float().norm(dim=(0, 2)).tolist() if state_grad is not None else []
    )
    if len(route_norms) != 8 or not all(math.isfinite(x) for x in route_norms):
        raise RuntimeError(f"{task}: invalid gradient at state routing boundary")
    expected = (
        range(4)
        if task == "classification"
        else range(4, 8)
        if task in ("segmentation", "sr")
        else range(8)
    )
    if step < 4 and not all(route_norms[i] > 0 for i in expected):
        raise RuntimeError(f"{task}: disconnected assigned slot")
    inactive = (
        range(4, 8)
        if task == "classification"
        else range(4)
        if task in ("segmentation", "sr")
        else []
    )
    if any(route_norms[i] != 0 for i in inactive):
        raise RuntimeError(f"{task}: decoder read the wrong state group")
    encoder_grad = math.sqrt(
        sum(
            float(p.grad.float().square().sum())
            for p in model.encoder.parameters()
            if p.requires_grad and p.grad is not None
        )
    )
    result.update(state_route_grad_norms=route_norms, encoder_grad_norm=encoder_grad)
    return result
