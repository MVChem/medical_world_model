"""EMA in adaptation-parameter space; only the target forward is detached."""
import math
import torch
from torch import nn

from .adaptation import shared_frozen_copy


class EMATarget(nn.Module):
    def __init__(self, online):
        super().__init__()
        self.parameter_names = tuple(n for n, p in online.named_parameters() if p.requires_grad)
        self.encoder = shared_frozen_copy(online).requires_grad_(False).eval()
        for module in self.encoder.modules():
            if hasattr(module, "gradient_checkpointing_disable"):
                module.gradient_checkpointing_disable()
        self.register_buffer("updates", torch.zeros((), dtype=torch.long))
        self.train(False)

    def train(self, mode=True):
        # Outer model.train() must not enable dropout or mutable target behavior.
        return super().train(False)

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    @torch.no_grad()
    def update(self, online, momentum):
        if not math.isfinite(momentum) or not 0 <= momentum < 1:
            raise ValueError("EMA momentum must be in [0, 1)")
        source, target = dict(online.named_parameters()), dict(self.encoder.named_parameters())
        actual = tuple(n for n, p in source.items() if p.requires_grad)
        if actual != self.parameter_names:
            raise ValueError("Online trainable parameter set changed after EMA initialization")
        for name in self.parameter_names:
            if target[name] is source[name] or target[name].requires_grad:
                raise RuntimeError("EMA mutable parameters must be independent and frozen")
            target[name].lerp_(source[name].detach().to(target[name].dtype), 1 - momentum)
        # Non-parameter buffers are state, not learned weights: copy them exactly.
        target_buffers = dict(self.encoder.named_buffers())
        for name, value in online.named_buffers():
            target_buffers[name].copy_(value)
        self.updates.add_(1)

    def compact_state(self):
        params = dict(self.encoder.named_parameters())
        return {"parameters": {n: params[n].detach().cpu().clone() for n in self.parameter_names},
                "buffers": {n: p.detach().cpu().clone() for n, p in self.encoder.named_buffers()},
                "updates": int(self.updates)}

    @torch.no_grad()
    def restore(self, state):
        if set(state) != {"parameters", "buffers", "updates"}:
            raise ValueError("Incomplete EMA state")
        params = dict(self.encoder.named_parameters())
        buffers = dict(self.encoder.named_buffers())
        if set(state["parameters"]) != set(self.parameter_names) or set(state["buffers"]) != set(buffers):
            raise ValueError("EMA checkpoint keys differ")
        if type(state["updates"]) is not int or state["updates"] < 0:
            raise ValueError("Invalid EMA update count")
        for saved, live in ((state["parameters"], params), (state["buffers"], buffers)):
            for name, value in saved.items():
                if live[name].shape != value.shape or live[name].dtype != value.dtype:
                    raise ValueError(f"EMA tensor mismatch: {name}")
                live[name].copy_(value)
        self.updates.fill_(state["updates"])
