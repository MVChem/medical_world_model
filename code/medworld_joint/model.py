"""Differentiable native VLM readout: four language and four vision-depth slots.

This is a new downstream adaptation pilot, initialized from local pretrained
weights. It does not resume the older V-JEPA world-model checkpoint. Both dense
tasks read all eight slots, permitting their losses to reach both VLM towers.
"""
import math
from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F

DENSE = Path(__file__).resolve().parents[1] / "medworld_dense_baselines"
if str(DENSE) not in sys.path:
    sys.path.insert(0, str(DENSE))
from common import digest, load_model_spec
from frozen_slots_extract import block_indices
from frozen_slots_train import FrozenSlotHead, positional_encoding

WIDTH = 1024
CONDITIONS = ("image_only", "joint_slots", "frozen_slots", "shuffled_slots", "joint_full_tokens")


class LoRALinear(nn.Module):
    """Frozen pretrained linear plus an explicitly differentiable FP32 adapter."""
    def __init__(self, base, rank=8, alpha=16):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x):
        reference = self.base(x)
        delta = F.linear(F.linear(x.to(self.lora_a.dtype), self.lora_a), self.lora_b)
        return reference + delta.to(reference.dtype) * self.scale


def inject_lora(module, names, rank):
    selected = []
    for path, child in list(module.named_modules()):
        if isinstance(child, nn.Linear) and path.rsplit(".", 1)[-1] in names:
            parent_path, _, leaf = path.rpartition(".")
            parent = module.get_submodule(parent_path) if parent_path else module
            setattr(parent, leaf, LoRALinear(child, rank, rank * 2))
            selected.append(path)
    if not selected:
        raise ValueError(f"no adapter targets matched {names}")
    return selected


class OnlineSlots(nn.Module):
    def __init__(self, mid, device="cuda", rank=8, language_layers=4,
                 vision_pixels=256, visual_tokens=64, checkpointing=True):
        super().__init__()
        from transformers import AutoImageProcessor, AutoTokenizer
        from transformers import Qwen3_5ForConditionalGeneration, Gemma3ForConditionalGeneration

        spec = load_model_spec(mid)
        if "fp8" in mid or mid == "medgemma27b":
            raise ValueError("27B joint adaptation requires a separate sharded/quantized training protocol")
        if digest(Path(spec["path"]) / "config.json") != spec["config_sha256"]:
            raise ValueError("local model config differs from registered checkpoint")
        cls = Qwen3_5ForConditionalGeneration if spec["family"] == "qwen" else Gemma3ForConditionalGeneration
        base = cls.from_pretrained(spec["path"], local_files_only=True,
                                   dtype=torch.bfloat16, attn_implementation="sdpa")
        base.requires_grad_(False)
        self.family = spec["family"]
        self.language = base.model.language_model
        self.vision = base.model.visual if self.family == "qwen" else base.model.vision_tower
        self.projector = None if self.family == "qwen" else base.model.multi_modal_projector
        language_width = int(base.config.text_config.hidden_size)
        vision_width = int(base.config.vision_config.hidden_size)
        del base
        self.processor = AutoImageProcessor.from_pretrained(spec["path"], local_files_only=True)
        if self.family == "qwen":
            self.processor.size = {"longest_edge": vision_pixels ** 2, "shortest_edge": vision_pixels ** 2}
            blocks = self.vision.blocks
        else:
            self.processor.do_pan_and_scan = False
            native_vision = getattr(self.vision, "vision_model", self.vision)
            blocks = native_vision.encoder.layers
        self.indices = block_indices(len(blocks))
        if language_layers != 4:
            raise ValueError("this protocol adapts exactly four sampled language depths")
        self.language_indices = block_indices(len(self.language.layers))
        language_adapters, vision_adapters = [], []
        for i in self.language_indices:
            targets = {"q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z",
                       "in_proj_b", "in_proj_a", "out_proj"}
            language_adapters.extend(f"layers.{i}.{name}" for name in
                                     inject_lora(self.language.layers[i], targets, rank))
        for i in self.indices:
            vision_adapters.extend(f"blocks.{i}.{name}" for name in
                                  inject_lora(blocks[i], {"qkv", "proj", "q_proj", "k_proj", "v_proj", "out_proj"}, rank))
        self.slot_queries = nn.Parameter(torch.randn(8, WIDTH) * .02)
        self.depth_project = nn.ModuleList([nn.Sequential(nn.LayerNorm(vision_width),
                                             nn.Linear(vision_width, WIDTH)) for _ in range(4)])
        self.language_query = nn.Linear(WIDTH, language_width)
        self.language_output = nn.ModuleList([nn.Sequential(nn.LayerNorm(language_width),
                                      nn.Linear(language_width, WIDTH)) for _ in range(4)])
        self.depth_output_norm = nn.LayerNorm(WIDTH)
        self.visual_tokens = visual_tokens
        self.captured = {}
        self.language_captured = {}
        self.handles = [blocks[i].register_forward_hook(self._capture(j)) for j, i in enumerate(self.indices)]
        self.handles += [self.language.layers[i].register_forward_hook(self._capture_language(j))
                         for j, i in enumerate(self.language_indices)]
        tokenizer = AutoTokenizer.from_pretrained(spec["path"], local_files_only=True)
        prompt = "Represent the visible chest radiograph for anatomical image analysis."
        ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")["input_ids"]
        self.register_buffer("prompt_ids", ids, persistent=False)
        if checkpointing:
            self.language.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.vision.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.to(device)
        self.metadata = dict(model_id=mid, model_label=spec["label"], family=self.family,
            checkpoint_path=spec["path"], checkpoint_config_sha256=spec["config_sha256"],
            checkpoint_files=spec["weight_files"], language_width=language_width, vision_width=vision_width,
            language_lora_modules=language_adapters, vision_lora_modules=vision_adapters,
            lora_rank=rank, language_lora_sampled_depths=self.language_indices, language_model_loaded=True,
            online_native_image_forward=True, cache_used=False, slot_count=8, slot_width=WIDTH,
            vision_depth_indices_zero_based=self.indices, language_visual_prefix_tokens=visual_tokens,
            language_prompt=prompt, patient_report_input=False, gradient_checkpointing=checkpointing,
            slot_sources=["one learned suffix query read at each sampled language depth"] * 4 +
                         ["learned attention query pooling at native vision depth"] * 4,
            dense_routing="all 8 slots (new pilot routing; older 4+4 routing is not reused)",
            language_positions="ordinary sequential positions over pooled native visual tokens, fixed prompt, four slot queries",
            processor=self.processor.to_dict())

    def _capture(self, index):
        def hook(_module, _args, output):
            self.captured[index] = output[0] if isinstance(output, tuple) else output
        return hook

    def _capture_language(self, index):
        def hook(_module, _args, output):
            self.language_captured[index] = output[0] if isinstance(output, tuple) else output
        return hook

    def forward_images(self, images, full_tokens=False):
        self.captured.clear()
        self.language_captured.clear()
        batch = len(images)
        reference = next(self.vision.parameters())
        inputs = self.processor(images=images, return_tensors="pt")
        pixels = inputs["pixel_values"].to(reference.device, dtype=reference.dtype)
        if self.family == "qwen":
            grid = inputs["image_grid_thw"]
            if grid.shape[0] != batch or not torch.equal(grid, grid[:1].expand_as(grid)):
                raise ValueError("batched Qwen depth pooling requires identical native image grids")
            outputs = self.vision(hidden_states=pixels,
                                  grid_thw=grid.to(reference.device))
            visual = outputs.pooler_output.reshape(batch, -1, outputs.pooler_output.shape[-1])
        else:
            if pixels.shape[0] != batch:
                raise ValueError("MedGemma processor unexpectedly produced multiple crops")
            outputs = self.vision(pixel_values=pixels)
            visual = self.projector(outputs.last_hidden_state)
        if set(self.captured) != set(range(4)):
            raise RuntimeError("missing native vision depth output")
        depth = []
        for i in ([3] if full_tokens else range(4)):
            native = self.captured[i]
            if native.ndim == 2:
                native = native.reshape(batch, -1, native.shape[-1])
            values = self.depth_project[i](native.float())
            if full_tokens:
                depth.append(values)
                continue
            query = self.slot_queries[4 + i]
            weights = (values * query).sum(-1).div(math.sqrt(WIDTH)).softmax(-1)
            depth.append(self.depth_output_norm((weights[..., None] * values).sum(1) + query))
        visual = F.adaptive_avg_pool1d(visual.transpose(1, 2), self.visual_tokens).transpose(1, 2)
        embed = self.language.get_input_embeddings()
        prompt = embed(self.prompt_ids).expand(batch, -1, -1)
        query = self.language_query(self.slot_queries[:4])[None].expand(batch, -1, -1).to(prompt.dtype)
        sequence = torch.cat([visual.to(prompt.dtype), prompt] + ([] if full_tokens else [query]), dim=1)
        mask = torch.ones(sequence.shape[:2], device=sequence.device, dtype=torch.long)
        positions = torch.arange(sequence.shape[1], device=sequence.device)[None].expand(batch, -1)
        language = self.language(inputs_embeds=sequence, attention_mask=mask, position_ids=positions,
                                  use_cache=False, return_dict=True).last_hidden_state
        if full_tokens:
            return torch.cat([self.language_output[-1](language[:, :self.visual_tokens].float()), depth[0]], dim=1)
        if set(self.language_captured) != set(range(4)):
            raise RuntimeError("missing language depth output")
        clinical = torch.stack([self.language_output[i](self.language_captured[i][:, -4+i].float())
                                for i in range(4)], dim=1)
        return torch.cat([clinical, torch.stack(depth, dim=1)], dim=1)

    def forward(self, images, full_tokens=False):
        # The prepared inputs are square canvases, giving equal native grids.
        # Verify that contract above before unpacking Qwen's concatenated tokens.
        return self.forward_images(images, full_tokens=full_tokens)


class EightSlotHead(FrozenSlotHead):
    """Same decoder family as the frozen baseline, with eight attached slots."""
    def __init__(self, task, full_tokens=False):
        super().__init__(task)
        self.full_tokens = full_tokens
        self.depth_position = positional_encoding(1, 8)

    def forward(self, image, slots):
        if slots.shape[-1] != WIDTH or (not self.full_tokens and slots.shape[1] != 8):
            raise ValueError(f"expected eight slots, got {tuple(slots.shape)}")
        z = F.interpolate(self.stem(image), (32, 32), mode="bilinear", align_corners=False)
        query = self.query_normalize(z.flatten(2).transpose(1, 2) + self.image_position)
        position = (positional_encoding(1, slots.shape[1]).to(slots.device)
                    if self.full_tokens else self.depth_position)
        values = self.slot_project(self.slot_normalize(slots.float())) + position
        attended, _ = self.attention(query, values, values, need_weights=False)
        attended = attended.transpose(1, 2).reshape(image.shape[0], 64, 32, 32)
        result = self.decode(self.fuse(torch.cat([z, attended], dim=1)))
        if self.task == "sr":
            return F.interpolate(image, scale_factor=4, mode="bicubic", align_corners=False) + .1 * result
        return result


class JointModel(nn.Module):
    def __init__(self, mid, task, condition, device="cuda", **encoder_args):
        super().__init__()
        if condition not in CONDITIONS:
            raise ValueError(condition)
        self.condition = condition
        # Construct the head before the encoder to preserve identical seeded
        # decoder initialization across image-only and encoder conditions.
        self.head = EightSlotHead(task, full_tokens=condition == "joint_full_tokens").to(device)
        self.encoder = None if condition == "image_only" else OnlineSlots(mid, device, **encoder_args)
        if condition == "frozen_slots":
            # Matched frozen-VLM control: train the same slot queries,
            # projections, and task decoder; only the backbone LoRA is frozen.
            for name, param in self.encoder.named_parameters():
                if ".lora_" in name:
                    param.requires_grad_(False)
        elif condition == "joint_full_tokens":
            self.encoder.slot_queries.requires_grad_(False)
            self.encoder.language_query.requires_grad_(False)
            self.encoder.depth_output_norm.requires_grad_(False)
            for modules in (self.encoder.depth_project, self.encoder.language_output):
                for module in list(modules)[:3]:
                    module.requires_grad_(False)
        self.last_slots = None

    def train(self, mode=True):
        super().train(mode)
        return self

    def forward(self, image, sources):
        if self.encoder is None:
            slots = image.new_zeros((len(image), 8, WIDTH))
        else:
            slots = self.encoder(sources, full_tokens=self.condition == "joint_full_tokens")
        self.last_slots = slots
        if torch.is_grad_enabled() and slots.requires_grad:
            slots.retain_grad()
        return self.head(image, slots)


def parameter_group(name):
    if name.startswith("head."):
        return "decoder"
    if name == "encoder.slot_queries":
        return "slot_queries"
    if ".lora_" in name:
        return "language_lora" if name.startswith("encoder.language.") else "vision_lora"
    return "slot_projection"


def gradient_audit(model):
    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            if param.grad is not None:
                raise AssertionError(f"frozen checkpoint weight received a gradient: {name}")
            continue
        group = parameter_group(name)
        value = float(param.grad.float().square().sum()) if param.grad is not None else 0.
        groups[group] = groups.get(group, 0.) + value
    result = dict(group_gradient_l2={key: math.sqrt(value) for key, value in groups.items()})
    if model.encoder is not None:
        if model.last_slots.grad is None:
            raise AssertionError("task loss did not reach the online VLM representation")
        if model.condition == "joint_full_tokens":
            for key in ("language_lora", "vision_lora"):
                value = result["group_gradient_l2"].get(key, 0.)
                if not math.isfinite(value) or value <= 0:
                    raise AssertionError(f"full-token loss did not reach {key}")
            result["full_token_count"] = model.last_slots.shape[1]
            return result
        if model.encoder.slot_queries.grad is None:
            raise AssertionError("task loss did not reach the online slot state or learned slot queries")
        result["slot_output_gradient_l2"] = model.last_slots.grad.float().square().sum((0, 2)).sqrt().tolist()
        result["slot_query_gradient_l2"] = model.encoder.slot_queries.grad.float().norm(dim=1).tolist()
        values = result["slot_output_gradient_l2"] + result["slot_query_gradient_l2"]
        if model.condition == "frozen_slots":
            result["frozen_vlm_lora_gradient_l2"] = 0.
            result["trainable_slot_readouts_with_frozen_vlm"] = True
        else:
            values += [result["group_gradient_l2"].get(key, 0.) for key in ("language_lora", "vision_lora")]
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise AssertionError(f"joint gradient flow was zero or nonfinite: {result}")
    return result


def adapter_state(model):
    return {name: param.detach().cpu().clone() for name, param in model.named_parameters() if param.requires_grad}


def load_adapter_state(model, state):
    expected = {name for name, param in model.named_parameters() if param.requires_grad}
    if set(state) != expected:
        raise ValueError("checkpoint trainable parameter names differ")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in state:
                param.copy_(state[name].to(param.device, dtype=param.dtype))


def update_audit(model, before):
    groups = {}
    for name, param in model.named_parameters():
        if name in before:
            value = float((param.detach().float().cpu() - before[name].float()).square().sum())
            group = parameter_group(name)
            groups[group] = groups.get(group, 0.) + value
    return {key: math.sqrt(value) for key, value in groups.items()}
