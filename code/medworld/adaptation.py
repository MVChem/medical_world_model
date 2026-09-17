"""LoRA and temporary depth capture, shared by the unified model only."""
from contextlib import contextmanager
import copy
import math

import torch
from torch import nn
import torch.nn.functional as F

LANGUAGE_TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv",
                    "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"}


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x):
        delta = F.linear(F.linear(x.to(self.lora_a.dtype), self.lora_a), self.lora_b)
        return self.base(x) + delta.to(x.dtype) * self.scale


def depths(count):
    if count < 4:
        raise ValueError("Four distinct readout depths are required")
    return [math.ceil((i + 1) * count / 4) - 1 for i in range(4)]


def adapt_selected(module, blocks, names, cfg):
    selected = []
    for i in depths(len(blocks)):
        for path, layer in list(blocks[i].named_modules()):
            if isinstance(layer, nn.Linear) and path.rsplit(".", 1)[-1] in names:
                parent, _, name = path.rpartition(".")
                owner = blocks[i].get_submodule(parent) if parent else blocks[i]
                setattr(owner, name, LoRALinear(layer, cfg["lora_rank"], cfg["lora_alpha"]))
                selected.append(f"{i}.{path}")
    if not selected:
        raise ValueError("No LoRA modules matched")
    module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return selected


def shared_frozen_copy(module):
    """Alias immutable Parameters; copy mutable Parameters and all buffers."""
    return copy.deepcopy(module, {id(p): p for p in module.parameters() if not p.requires_grad})


@contextmanager
def capture_depths(blocks):
    values, handles = {}, []
    for j, i in enumerate(depths(len(blocks))):
        def capture(_module, _args, output, j=j):
            values[j] = output[0] if isinstance(output, tuple) else output
        handles.append(blocks[i].register_forward_hook(capture))
    try:
        yield values
    finally:
        for handle in handles:
            handle.remove()
