"""Four-task Stage-1 prototype with JEPA fusion and native-vision 4+4 readouts.

This module has no patient-report input and no temporal predictor. A single
language model (including its LoRA) encodes image evidence and decodes reports.
Spatial tasks read only native visual tokens, with LR-only inputs for SR.
Smoke results are not eligible six-task paper results.
"""
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F

CODE = Path(__file__).resolve().parents[1]
PROJECT = CODE.parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
from medworld_joint.model import OnlineSlots
from medworld_common.qwen import chunked_ce
from medworld_dense_baselines.frozen_slots_train import FrozenSlotHead, positional_encoding, objective

TASKS = ("classification", "report", "segmentation", "sr")
CONDITIONS = ("slots", "full_tokens", "shuffled", "image_only")
WIDTH = 1024


class ClassificationHead(nn.Module):
    def __init__(self, findings=13):
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(WIDTH), nn.Linear(WIDTH, 128))
        self.queries = nn.Parameter(torch.randn(findings, 128) * .02)
        self.attention = nn.MultiheadAttention(128, 4, batch_first=True, dropout=0)
        self.output = nn.Linear(128, 1)

    def forward(self, state):
        values = self.project(state.float())
        queries = self.queries[None].expand(len(state), -1, -1)
        readout, _ = self.attention(queries, values, values, need_weights=False)
        return self.output(queries + readout).squeeze(-1)


class SpatialHead(FrozenSlotHead):
    """Retain the matched dense decoder while allowing gradients into context."""
    def forward(self, image, context):
        z = F.interpolate(self.stem(image), (32, 32), mode="bilinear", align_corners=False)
        queries = self.query_normalize(z.flatten(2).transpose(1, 2) + self.image_position)
        position = positional_encoding(1, context.shape[1]).to(context.device)
        values = self.slot_project(self.slot_normalize(context.float())) + position
        attended, _ = self.attention(queries, values, values, need_weights=False)
        attended = attended.transpose(1, 2).reshape(len(image), 64, 32, 32)
        prediction = self.decode(self.fuse(torch.cat([z, attended], 1)))
        if self.task == "sr":
            prediction = F.interpolate(image, scale_factor=4, mode="bicubic", align_corners=False) + .1 * prediction
        return prediction


class StateEncoder(OnlineSlots):
    def __init__(self, model_id, device, rank=8, vision_pixels=256):
        super().__init__(model_id, device=device, rank=rank, vision_pixels=vision_pixels)
        language_width = self.language.get_input_embeddings().weight.shape[1]
        self.jepa_adapter = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, language_width), nn.GELU(),
                                          nn.Linear(language_width, language_width)).to(device)
        self.jepa_positions = nn.Parameter(torch.randn(1, 64, language_width, device=device) * .02)
        if str(CODE / "vjepa2") not in sys.path:
            sys.path.insert(0, str(CODE / "vjepa2"))
        from app.vjepa_2_1.models.vision_transformer import vit_base
        self.jepa = vit_base(img_size=(384, 384), patch_size=16, num_frames=64, tubelet_size=2,
                             use_sdpa=True, use_rope=True, img_temporal_dim_size=1, interpolate_rope=True)
        self.jepa_checkpoint = PROJECT / "code/medworld_table1/weights/vjepa2_1_vitb.pt"
        saved = torch.load(self.jepa_checkpoint, map_location="cpu", weights_only=True)["ema_encoder"]
        self.jepa.load_state_dict({k.replace("module.", "").replace("backbone.", ""): v
                                   for k, v in saved.items()}, strict=True)
        self.jepa.requires_grad_(False).eval().to(device=device, dtype=torch.bfloat16)
        self.last_state = None
        self.metadata.update(fusion_input="frozen V-JEPA 2.1 ViT-B 384px, 24x24 pooled to8x8, trainable adapter",
                             jepa_checkpoint=str(self.jepa_checkpoint),
                             dense_routing="only four native visual slots (or all final native visual tokens)",
                             patient_report_input=False, temporal_predictor_used=False,
                             frozen_vjepa_features_used=True, cache_used=False)

    def train(self, mode=True):
        super().train(mode)
        self.jepa.eval()
        return self

    def visual_state(self, images, full_tokens=False):
        self.captured.clear()
        reference = next(self.vision.parameters())
        inputs = self.processor(images=images, return_tensors="pt")
        pixels = inputs["pixel_values"].to(reference.device, dtype=reference.dtype)
        grid = inputs["image_grid_thw"]
        if not torch.equal(grid, grid[:1].expand_as(grid)):
            raise ValueError("batch requires matching prepared square image grids")
        self.vision(hidden_states=pixels, grid_thw=grid.to(reference.device))
        state = []
        for i in ([3] if full_tokens else range(4)):
            raw = self.captured[i]
            if raw.ndim == 2:
                raw = raw.reshape(len(images), -1, raw.shape[-1])
            values = self.depth_project[i](raw.float())
            if full_tokens:
                state.append(values)
            else:
                query = self.slot_queries[4 + i]
                weights = (values * query).sum(-1).div(math.sqrt(WIDTH)).softmax(-1)
                state.append(self.depth_output_norm((weights[..., None] * values).sum(1) + query))
        return torch.cat(state, 1) if full_tokens else torch.stack(state, 1)

    def fusion_state(self, images, full_tokens=False):
        device = next(self.language.parameters()).device
        pixels = torch.stack([torch.from_numpy(np.array(im.convert("RGB").resize((384, 384),
                    Image.Resampling.BICUBIC), copy=True)).permute(2, 0, 1) for im in images]).to(device).float() / 255
        mean = pixels.new_tensor([.485, .456, .406])[None, :, None, None]
        std = pixels.new_tensor([.229, .224, .225])[None, :, None, None]
        with torch.no_grad():
            features = self.jepa(((pixels - mean) / std).unsqueeze(2))
            if features.shape[1:] != (576, 768):
                raise ValueError(f"unexpected JEPA shape {features.shape}")
            features = F.adaptive_avg_pool2d(features.transpose(1, 2).reshape(-1, 768, 24, 24), (8, 8))
            features = features.flatten(2).transpose(1, 2)
        visual = self.jepa_adapter(features.float()) + self.jepa_positions
        embed = self.language.get_input_embeddings()
        prompt = embed(self.prompt_ids)[None] if self.prompt_ids.ndim == 1 else embed(self.prompt_ids)
        prompt = prompt.expand(len(images), -1, -1)
        parts = [visual.to(prompt.dtype), prompt]
        if not full_tokens:
            parts.append(self.language_query(self.slot_queries[:4])[None].expand(len(images), -1, -1).to(prompt.dtype))
        sequence = torch.cat(parts, 1)
        mask = torch.ones(sequence.shape[:2], dtype=torch.long, device=device)
        positions = torch.arange(sequence.shape[1], device=device)[None].expand(len(images), -1)
        self.language_captured.clear()
        output = self.language(inputs_embeds=sequence, attention_mask=mask, position_ids=positions,
                               use_cache=False, return_dict=True).last_hidden_state
        if full_tokens:
            return self.language_output[-1](output.float())
        return torch.stack([self.language_output[i](self.language_captured[i][:, -4+i].float())
                            for i in range(4)], 1)

    def encode(self, images, spatial=False, full_tokens=False):
        visual = self.visual_state(images, full_tokens)
        state = visual if spatial else torch.cat([self.fusion_state(images, full_tokens), visual], 1)
        if state.requires_grad:
            state.retain_grad()
        self.last_state = state
        return state


class MultiTaskModel(nn.Module):
    def __init__(self, model_id="qwen08b", condition="slots", device="cuda", rank=8, vision_pixels=256,
                 report_tokens=384):
        super().__init__()
        if condition not in CONDITIONS:
            raise ValueError(condition)
        if model_id not in ("qwen08b", "qwen9b"):
            raise ValueError("first validate Qwen 0.8B, then Qwen 9B")
        self.condition, self.device = condition, torch.device(device)
        self.report_tokens = report_tokens
        # Construct task heads before any condition-dependent module creation.
        self.classification = ClassificationHead()
        self.register_buffer("pos_weight", torch.ones(13))
        self.segmentation = SpatialHead("segmentation")
        self.sr = SpatialHead("sr")
        self.encoder = None if condition == "image_only" else StateEncoder(model_id, device, rank, vision_pixels)
        self.tokenizer = None
        self.output_head = None
        self.metadata = dict(model_id=model_id, condition=condition, rank=rank, vision_pixels=vision_pixels,
                             report_tokens=report_tokens, stage=1, temporal_predictor_used=False,
                             tasks=list(TASKS) if self.encoder is not None else ["segmentation", "sr"],
                             pending_tasks=["official_vqa", "ms_cxr_grounding"],
                             patient_report_input=False, sr_inputs="both branches from LR only",
                             shared_checkpoint_across_tasks=True, dense_training_cohort=4096)
        if self.encoder is not None:
            from transformers import AutoTokenizer
            path = Path(self.encoder.metadata["checkpoint_path"])
            self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
            self.tokenizer.padding_side = "right"
            cfg = json.loads((path / "config.json").read_text())
            width = self.encoder.language.get_input_embeddings().weight.shape[1]
            self.report_projection = nn.Sequential(nn.LayerNorm(WIDTH), nn.Linear(WIDTH, width))
            if not cfg.get("tie_word_embeddings", cfg.get("text_config", {}).get("tie_word_embeddings", False)):
                from safetensors import safe_open
                index = path / "model.safetensors.index.json"
                shard = json.loads(index.read_text())["weight_map"]["lm_head.weight"] if index.exists() else "model.safetensors"
                with safe_open(path / shard, framework="pt", device="cpu") as f:
                    weight = f.get_tensor("lm_head.weight")
                self.output_head = nn.Linear(width, weight.shape[0], bias=False, dtype=weight.dtype)
                self.output_head.weight = nn.Parameter(weight, requires_grad=False)
            prompt = self.tokenizer.apply_chat_template([
                {"role": "system", "content": "Write a concise chest radiograph report with FINDINGS and IMPRESSION. Use the supplied image state."},
                {"role": "user", "content": "Describe the supplied image state."}],
                tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)
            self.register_buffer("report_prompt", torch.tensor(prompt, dtype=torch.long), persistent=False)
            if condition == "full_tokens":
                self.encoder.slot_queries.requires_grad_(False)
                self.encoder.language_query.requires_grad_(False)
            self.metadata.update(encoder=self.encoder.metadata, report_decoder="shared Qwen language backbone and LoRA",
                                 output_head="pretrained untied" if self.output_head is not None else "tied input embeddings")
        self.to(device)

    def set_pos_weight(self, value):
        weight = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        if weight.shape != (13,) or not torch.isfinite(weight).all() or (weight <= 0).any():
            raise ValueError("classification weights must be finite positive values for 13 findings")
        self.pos_weight.copy_(weight)
        self.metadata["classification_pos_weight"] = weight.cpu().tolist()

    def gradient_contract(self, task):
        """Check the actual task-to-branch routing after backward."""
        if self.encoder is None:
            return {"encoder_absent": True}
        groups = {"language_lora": [], "vision_lora": [], "jepa_adapter": []}
        for name, parameter in self.encoder.named_parameters():
            if parameter.grad is None:
                continue
            key = ("language_lora" if name.startswith("language.") and "lora_" in name else
                   "vision_lora" if name.startswith("vision.") and "lora_" in name else
                   "jepa_adapter" if name.startswith("jepa_adapter.") else None)
            if key:
                groups[key].append(parameter.grad.detach().float().square().sum())
        norms = {key: float(torch.stack(values).sum().sqrt()) if values else 0.
                 for key, values in groups.items()}
        spatial = task in ("segmentation", "sr")
        for key, value in norms.items():
            expected = key == "vision_lora" or not spatial
            if not math.isfinite(value) or (expected and value <= 0) or (not expected and value != 0):
                raise RuntimeError(f"{task} gradient routing failed for {key}: {value}")
        if self.condition != "full_tokens":
            grad = self.encoder.slot_queries.grad
            if grad is None:
                raise RuntimeError(f"{task} did not reach slot queries")
            queries = grad.detach().float().norm(dim=-1)
            expected = torch.ones(8, device=queries.device, dtype=torch.bool)
            if spatial:
                expected[:4] = False
            if not torch.isfinite(queries).all() or not (queries[expected] > 0).all() or (queries[~expected] != 0).any():
                raise RuntimeError(f"{task} slot gradient routing failed: {queries.tolist()}")
            norms["slot_query_gradient_l2"] = queries.tolist()
        return norms

    def state(self, task, batch):
        if self.encoder is None:
            if task not in ("segmentation", "sr"):
                raise ValueError("image-only is a spatial decoder control")
            return torch.zeros(len(batch["images"]), 4, WIDTH, device=self.device)
        images = batch["images"]
        if self.condition == "shuffled":
            if "donor_images" not in batch:
                raise ValueError("shuffled state requires explicit same-split, different-patient donors")
            images = batch["donor_images"]
        return self.encoder.encode(images, spatial=task in ("segmentation", "sr"),
                                   full_tokens=self.condition == "full_tokens")

    def report_prefix(self, state):
        embed = self.encoder.language.get_input_embeddings()
        prompt = embed(self.report_prompt)[None].expand(len(state), -1, -1)
        return torch.cat([self.report_projection(state.float()).to(prompt.dtype), prompt], 1)

    def language_weight(self):
        return self.output_head.weight if self.output_head is not None else self.encoder.language.get_input_embeddings().weight

    def loss(self, task, batch):
        state = self.state(task, batch)
        if task == "classification":
            logits = self.classification(state).float()
            labels = batch["labels"].to(self.device).float()
            mask = batch["label_mask"].to(self.device)
            loss = F.binary_cross_entropy_with_logits(logits, labels.clamp(0, 1), pos_weight=self.pos_weight,
                                                      reduction="none")
            counts = mask.sum(0)
            return ((loss * mask).sum(0) / counts.clamp_min(1))[counts > 0].mean()
        if task in ("segmentation", "sr"):
            output = getattr(self, task)(batch["pixels"].to(self.device), state)
            return objective(task, output, batch["targets"].to(self.device), batch["mask"].to(self.device))
        if task != "report":
            raise ValueError(task)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        encoded = self.tokenizer(batch["report_targets"], return_tensors="pt", padding=True,
                                 truncation=True, max_length=self.report_tokens - 1, add_special_tokens=False)
        ids, valid = encoded["input_ids"].to(self.device), encoded["attention_mask"].to(self.device)
        # Append EOS at each sample's actual target end, before batch padding.
        lengths = valid.sum(1)
        ids = F.pad(ids, (0, 1), value=self.tokenizer.pad_token_id)
        valid = F.pad(valid, (0, 1))
        ids[torch.arange(len(ids), device=self.device), lengths] = self.tokenizer.eos_token_id
        valid[torch.arange(len(ids), device=self.device), lengths] = 1
        prefix = self.report_prefix(state)
        sequence = torch.cat([prefix, self.encoder.language.get_input_embeddings()(ids)], 1)
        mask = torch.cat([torch.ones(prefix.shape[:2], device=self.device, dtype=torch.long), valid], 1)
        positions = torch.arange(sequence.shape[1], device=self.device)[None].expand(len(ids), -1)
        output = self.encoder.language(inputs_embeds=sequence, attention_mask=mask, position_ids=positions,
                                       use_cache=False, return_dict=True).last_hidden_state
        return chunked_ce(output[:, prefix.shape[1]-1:-1], ids.masked_fill(~valid.bool(), -100),
                          self.language_weight(), chunk=32)

    @torch.no_grad()
    def predict(self, task, batch, max_new_tokens=384):
        state = self.state(task, batch)
        if task == "classification":
            return self.classification(state)
        if task in ("segmentation", "sr"):
            return getattr(self, task)(batch["pixels"].to(self.device), state)
        if task != "report":
            raise ValueError(task)
        sequence = self.report_prefix(state)
        ended = torch.zeros(len(state), dtype=torch.bool, device=self.device)
        generated = []
        # Short smoke decoding deliberately avoids KV-cache/checkpointing state.
        # Formal report scoring should profile this shared-backbone decoder.
        for _ in range(max_new_tokens):
            mask = torch.ones(sequence.shape[:2], device=self.device, dtype=torch.long)
            positions = torch.arange(sequence.shape[1], device=self.device)[None].expand(len(state), -1)
            output = self.encoder.language(inputs_embeds=sequence, attention_mask=mask, position_ids=positions,
                                           use_cache=False, return_dict=True).last_hidden_state
            weight = self.language_weight()
            token = F.linear(output[:, -1].to(weight.dtype), weight).argmax(-1)
            token = torch.where(ended, self.tokenizer.eos_token_id, token)
            generated.append(token)
            ended |= token == self.tokenizer.eos_token_id
            if ended.all():
                break
            sequence = torch.cat([sequence, self.encoder.language.get_input_embeddings()(token[:, None])], 1)
        return self.tokenizer.batch_decode(torch.stack(generated, 1).cpu(), skip_special_tokens=True)
