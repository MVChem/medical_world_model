"""Raw-input task model with an optional, independent slot-producing branch."""
import hashlib
import torch
from torch import nn
import torch.nn.functional as F

from .adaptation import depths
from .architecture import architecture, normalize_baseline, uses_slot_branch, uses_temporal
from .downstream_tasks.classification import ClassificationHead
from .downstream_tasks.text import TextDecoder
from .downstream_tasks.segmentation.decoder import SegmentationHead
from .downstream_tasks.training import current_loss as current_task_loss
from .downstream_tasks.registry import TASKS
from .downstream_tasks.common.decoder import TaskDecoder
from .ema import EMATarget
from .encoder import FrozenJEPA, StateEncoder
from .predictor import WorldModel


class MedWorld(nn.Module):
    def __init__(self, cfg, device="cuda"):
        super().__init__()
        self.cfg = normalize_baseline(dict(cfg))
        architecture(self.cfg)
        cfg = self.cfg
        # Common modules have identical initial tensors regardless of the slot
        # switch, pretrained model size, or RNG consumed loading those assets.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg["seed"] + 2048)
            self.task_decoder = TaskDecoder(cfg)
            self.classification = ClassificationHead(width=cfg["decoder_width"])
            self.segmentation = SegmentationHead(width=cfg["decoder_width"], channels=cfg["segmentation_channels"])
            self.text = TextDecoder(cfg)
        digest = hashlib.sha256()
        for name, value in self.named_parameters():
            digest.update(name.encode())
            digest.update(str((tuple(value.shape), value.dtype)).encode())
            digest.update(value.detach().cpu().numpy().tobytes())
        self.register_buffer("pos_weight", torch.ones(13))
        self.target = None
        self.metadata = {
            "architecture": "raw_input_v1", "slot_conditioning": cfg["slot_conditioning"],
            "task_initialization_sha256": digest.hexdigest(),
            "task_inputs": ["raw image patches", "optional supplied report"],
            "text_decoder_inputs": ["task features", "raw question"],
            "text_decoder": "independent byte autoregressive Transformer; trained from scratch",
            "text_length_unit": "UTF-8 bytes",
            "task_pretrained_backbone": None,
            "current_dataset_inputs": ["image", "VQA question"],
            "slot_branch": uses_slot_branch(cfg),
            "state_shape": [8, 1024] if uses_slot_branch(cfg) else None,
            "current_tasks": list(TASKS),
            "segmentation_channels": cfg["segmentation_channels"],
            "report_only_input": not uses_slot_branch(cfg), "ehr_input": False,
            "frozen_backbone_shared": False, "encoder_decoder_lora_shared": False,
            "training": "current task loss" + (" plus temporal slot prediction" if uses_temporal(cfg) else ""),
            "representation_objective": "visual_slot_spatial_consistency" if cfg.get("visual_consistency_weight", 0) else None,
            "latent_weight": cfg["latent_weight"],
            "visual_consistency_weight": cfg["visual_consistency_weight"],
            "target_update": "EMA after optimizer updates" if uses_temporal(cfg) else None,
        }
        if uses_slot_branch(cfg):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(cfg["seed"] + 4096)
                self._init_slot_branch()
        self.to(device)
        if uses_temporal(cfg):
            self.target = EMATarget(self.encoder).to(self.device)

    def _init_slot_branch(self):
        """Load pretrained assets only for the model that supplies slot conditions."""
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
        from transformers.models.qwen3_5 import modeling_qwen3_5
        cfg = self.cfg
        fast_kernels = modeling_qwen3_5.is_fast_path_available
        if cfg.get("require_fast_kernels") and not fast_kernels:
            raise RuntimeError("Fast Qwen kernels required: install flash-linear-attention and causal-conv1d")
        base = Qwen3_5ForConditionalGeneration.from_pretrained(
            cfg["qwen"], local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa")
        base.requires_grad_(False)
        if cfg.get("batched_vision_attention"):
            from .vision_attention import batch_vision_attention
            batch_vision_attention(base.model.visual)
        if cfg.get("visual_consistency_weight", 0) > 0:
            from .representation import FrozenFeatureTeacher, SlotSpatialReconstruction
            self.visual_teacher = FrozenFeatureTeacher(base.model.visual)
            self.visual_reconstruction = SlotSpatialReconstruction()
        self.encoder = StateEncoder(base.model.language_model, base.model.visual, cfg)
        self.processor = AutoProcessor.from_pretrained(cfg["qwen"], local_files_only=True)
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.jepa = FrozenJEPA(cfg["jepa"])
        if uses_temporal(cfg):
            self.world = WorldModel(cfg)
        self.metadata.update(
            qwen_fast_kernels=fast_kernels,
            language_depths_1based=[i + 1 for i in depths(len(self.encoder.language.layers))],
            vision_depths_1based=[i + 1 for i in depths(len(self.encoder.vision.blocks))],
            temporal_inputs=["source image", "source report", "signed delta_hours"],
            time_condition="signed realized_gap_hours")

    @property
    def device(self):
        return self.pos_weight.device

    def forward(self, task, batch, temporal_batch=None):
        """One DDP forward owns current-task and temporal graphs for an update."""
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                            enabled=self.cfg.get("amp", False)):
            if task == "temporal":
                if temporal_batch is not None:
                    raise ValueError("Joint updates require a current task")
                return self.temporal_loss(batch)
            loss, parts = self.current_loss(task, batch)
            if temporal_batch is not None:
                temporal, temporal_parts = self.temporal_loss(temporal_batch)
                loss = loss + temporal
                parts.update({"temporal_" + key: value for key, value in temporal_parts.items()})
        return loss, parts

    def prepare_observation(self, images, *, slot_only=False):
        """CPU-only, exact preprocessing for the bounded batch prefetch worker."""
        prepared = {}
        if not slot_only and images is not None:
            prepared["task"] = self.task_decoder.prepare_images(images)
        if not uses_slot_branch(self.cfg):
            return prepared
        pixels = self.cfg["vision_pixels"]
        prepared.update({"vision": self.processor.image_processor(
                    images=images, return_tensors="pt", min_pixels=pixels**2, max_pixels=pixels**2),
                "jepa": self.jepa.prepare_pixels(images)})
        return prepared

    def prepare_batch(self, task, batch):
        if task == "temporal":
            if not uses_temporal(self.cfg):
                raise ValueError("This model has no temporal training objective")
            for side in ("source", "target"):
                batch[side]["prepared"] = self.prepare_observation(batch[side]["images"], slot_only=True)
        else:
            batch["prepared"] = self.prepare_observation(batch.get("images"))
        return batch

    def encode(self, images, texts=None, *, target=False, prepared=None):
        if not uses_slot_branch(self.cfg):
            raise ValueError("Raw-input baseline has no slot encoder")
        if not images:
            raise ValueError("At least one observation image is required")
        encoder = self.target if target else self.encoder
        if encoder is None:
            raise ValueError("This model has no temporal target encoder")
        pixels = self.cfg["vision_pixels"]
        inputs = prepared["vision"] if prepared is not None else self.processor.image_processor(
            images=images, return_tensors="pt", min_pixels=pixels**2, max_pixels=pixels**2)
        inputs = {k: v.to(self.device) for k, v in inputs.items() if torch.is_tensor(v)}
        if texts is None:
            texts = [""] * len(images)
        if len(texts) != len(images) or any(not isinstance(t, str) for t in texts):
            raise ValueError("Expected one observation text per image")
        # Identical instruction at every time point. The encoder never sees the
        # requested direction, time gap, or the other observation's evidence.
        texts = [self.cfg["observation_prompt"] +
                 ("\nReport:\n" + t if t else "") for t in texts]
        tokens = self.tokenizer(texts, padding=True, truncation=True,
                                max_length=self.cfg["context_tokens"], return_tensors="pt")
        features = self.jepa(images, prepared_pixels=prepared["jepa"]) if prepared is not None else self.jepa(images)
        return encoder(inputs, tokens["input_ids"].to(self.device),
                       tokens["attention_mask"].to(self.device), features)

    def task_inputs(self, images=None, prepared=None, reports=None):
        """Original task input plus optional slots, with no shared VLM features."""
        slots = None
        if self.cfg["slot_conditioning"]:
            if images is None or len(images) == 0:
                raise ValueError("The eight-slot encoder requires images; report-only slots are not defined")
            slots = self.encode(images, texts=reports, prepared=prepared)
        features = self.task_decoder(images=images, reports=reports, slots=slots,
                                     prepared=prepared.get("task") if prepared is not None else None)
        return features, slots

    def current_loss(self, task, batch):
        loss, parts, state = current_task_loss(self, task, batch, return_state=True)
        if (self.training and task == "segmentation" and state is not None
                and hasattr(self, "visual_reconstruction")):
            auxiliary = self.visual_reconstruction.loss(
                state[:, 4:], batch["pixels"].to(self.device), batch["mask"].amax(dim=1, keepdim=True).to(self.device),
                batch["ids"], self.visual_teacher, self.processor, self.cfg)
            weighted = self.cfg["visual_consistency_weight"] * auxiliary
            parts.update(visual_consistency=auxiliary.detach(), visual_consistency_weighted=weighted.detach())
            loss = loss + weighted
        return loss, parts

    def temporal_loss(self, batch):
        if not uses_temporal(self.cfg):
            raise ValueError("This model has no temporal training objective")
        # Representation learning only; no temporal classification/report task.
        target = self.encode(**batch["target"], target=True)
        source = self.encode(**batch["source"])
        predicted = self.world(source, batch["delta_hours"].to(self.device))
        latent = F.mse_loss(F.layer_norm(predicted.float(), (1024,)),
                            F.layer_norm(target.float(), (1024,)))
        return self.cfg["latent_weight"] * latent, {"latent": latent.detach()}

    @torch.no_grad()
    def predict(self, task, batch):
        if task == "segmentation" and not batch.get("images"):
            raise ValueError("Segmentation requires an input image with spatial coordinates")
        features, slots = self.task_inputs(batch.get("images"), reports=batch.get("reports"))
        if task == "classification":
            return self.classification(features).sigmoid()
        if task == "segmentation":
            return self.segmentation(features)
        if task == "vqa":
            return self.text.generate(features, batch["questions"], self.cfg["generation_tokens"])
        raise ValueError(f"Unknown task: {task}")

    def compact_state(self):
        return {
            "parameters": {n: p.detach().cpu().clone() for n, p in self.named_parameters() if p.requires_grad},
            "pos_weight": self.pos_weight.detach().cpu().clone(),
            "ema": self.target.compact_state() if self.target is not None else None,
        }

    @torch.no_grad()
    def restore(self, state):
        expected = {n: p for n, p in self.named_parameters() if p.requires_grad}
        if set(state) != {"parameters", "pos_weight", "ema"} or set(state["parameters"]) != set(expected):
            raise ValueError("Unified checkpoint parameter keys differ")
        for name, value in state["parameters"].items():
            if expected[name].shape != value.shape or expected[name].dtype != value.dtype:
                raise ValueError(f"Checkpoint tensor mismatch: {name}")
            expected[name].copy_(value)
        if state["pos_weight"].shape != self.pos_weight.shape:
            raise ValueError("Classification weights differ")
        self.pos_weight.copy_(state["pos_weight"])
        if self.target is None:
            if state["ema"] is not None:
                raise ValueError("Task-only checkpoint must not contain EMA state")
        else:
            if state["ema"] is None:
                raise ValueError("Temporal checkpoints require EMA state")
            self.target.restore(state["ema"])
