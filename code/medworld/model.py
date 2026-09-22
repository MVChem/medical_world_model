"""Unified current-task and signed-time model. No observation bypass in text decoding."""
import torch
from torch import nn
import torch.nn.functional as F

from .adaptation import LANGUAGE_TARGETS, adapt_selected, depths, shared_frozen_copy
from .downstream_tasks.classification import ClassificationHead, finding_loss
from .downstream_tasks.text import TextDecoder
from .downstream_tasks.segmentation.decoder import SegmentationHead
from .downstream_tasks.training import current_loss as current_task_loss
from .downstream_tasks.registry import TASKS
from .downstream_tasks.common.decoder import TaskDecoder
from .ema import EMATarget
from .encoder import FrozenJEPA, StateEncoder
from .predictor import WorldModel

TEMPORAL_COLUMNS = (0, 1, 2, 3, 8, 11)


class MedWorld(nn.Module):
    def __init__(self, cfg, device="cuda"):
        super().__init__()
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
        self.cfg = dict(cfg)
        from transformers.models.qwen3_5 import modeling_qwen3_5
        fast_kernels = modeling_qwen3_5.is_fast_path_available
        if cfg.get("require_fast_kernels") and not fast_kernels:
            raise RuntimeError("Fast Qwen kernels required: install flash-linear-attention and causal-conv1d")
        base = Qwen3_5ForConditionalGeneration.from_pretrained(
            cfg["qwen"], local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa")
        base.requires_grad_(False)
        if cfg.get("batched_vision_attention"):
            from .vision_attention import batch_vision_attention
            batch_vision_attention(base.model.visual)
        # Clone the module tree before adding LoRA. Immutable pretrained tensors
        # remain shared, while encoder and decoder adapters are independent.
        decoder_language = shared_frozen_copy(base.model.language_model)
        if cfg.get("visual_consistency_weight", 0) > 0:
            from .representation import FrozenFeatureTeacher
            self.visual_teacher = FrozenFeatureTeacher(base.model.visual)
        self.encoder = StateEncoder(base.model.language_model, base.model.visual, cfg)
        adapt_selected(decoder_language, decoder_language.layers, LANGUAGE_TARGETS, cfg)
        self.processor = AutoProcessor.from_pretrained(cfg["qwen"], local_files_only=True)
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "left"
        if self.tokenizer.eos_token_id is None:
            raise ValueError("Report decoding requires an EOS token")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.text = TextDecoder(decoder_language, base.lm_head, self.tokenizer, cfg)
        self.jepa = FrozenJEPA(cfg["jepa"])
        self.world = WorldModel(cfg)
        self.task_decoder = TaskDecoder(base.lm_head.weight.shape[1], cfg["decoder_width"], cfg["decoder_depth"])
        self.classification = ClassificationHead(width=cfg["decoder_width"])
        self.segmentation = SegmentationHead(width=cfg["decoder_width"], channels=cfg.get("segmentation_channels", 3))
        if cfg.get("visual_consistency_weight", 0) > 0:
            from .representation import SlotSpatialReconstruction
            # Do not perturb decoder initialization or the training RNG stream.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(cfg["seed"] + 9138)
                self.visual_reconstruction = SlotSpatialReconstruction()
        self.register_buffer("pos_weight", torch.ones(13))
        self.metadata = {
            "qwen_fast_kernels": fast_kernels,
            "batched_vision_attention": cfg.get("batched_vision_attention", False),
            "representation_objective": "visual_slot_spatial_consistency" if hasattr(self, "visual_reconstruction") else None,
            "training": "joint current-task and temporal loss from the first update",
            "architecture": "medworld_image_conditioned_v3", "slot_conditioning": cfg["slot_conditioning"], "state_shape": [8, 1024],
            "language_depths_1based": [i + 1 for i in depths(len(self.encoder.language.layers))],
            "vision_depths_1based": [i + 1 for i in depths(len(self.encoder.vision.blocks))],
            "target_update": "EMA of all trainable state encoder parameters after each joint optimizer update",
            "text_decoder_inputs": ["image_features", "question", "optional_eight_slots"],
            "time_condition": "signed realized_gap_hours; direction is a predictor input only",
            "current_tasks": list(TASKS),
            "segmentation_channels": cfg.get("segmentation_channels", 3),
            "temporal_inputs": ["source image", "source report", "signed delta_hours"],
            "report_only_input": False, "ehr_input": False,
            "frozen_backbone_shared": True, "encoder_decoder_lora_shared": False,
        }
        self.to(device)
        self.target = EMATarget(self.encoder).to(self.device)

    @property
    def device(self):
        return self.encoder.slot_queries.device

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

    def prepare_observation(self, images):
        """CPU-only, exact preprocessing for the bounded batch prefetch worker."""
        pixels = self.cfg["vision_pixels"]
        return {"vision": self.processor.image_processor(
                    images=images, return_tensors="pt", min_pixels=pixels**2, max_pixels=pixels**2),
                "jepa": self.jepa.prepare_pixels(images)}

    def prepare_batch(self, task, batch):
        if task == "temporal":
            for side in ("source", "target"):
                batch[side]["prepared"] = self.prepare_observation(batch[side]["images"])
        else:
            batch["prepared"] = self.prepare_observation(batch["images"])
        return batch

    def encode(self, images, texts=None, *, spatial=False, target=False, return_image_features=False, prepared=None):
        if not images:
            raise ValueError("At least one observation image is required")
        encoder = self.target if target else self.encoder
        pixels = self.cfg["vision_pixels"]
        inputs = prepared["vision"] if prepared is not None else self.processor.image_processor(
            images=images, return_tensors="pt", min_pixels=pixels**2, max_pixels=pixels**2)
        inputs = {k: v.to(self.device) for k, v in inputs.items() if torch.is_tensor(v)}
        if spatial:
            return encoder(inputs, spatial=True)
        if texts is None:
            texts = [""] * len(images)
        if len(texts) != len(images) or any(not isinstance(t, str) for t in texts):
            raise ValueError("Expected one observation text per image")
        # Identical instruction at every time point. The encoder never sees the
        # requested direction, time gap, or the other observation's evidence.
        texts = [self.cfg.get("observation_prompt", "Chest radiograph observation.") +
                 ("\nReport:\n" + t if t else "") for t in texts]
        tokens = self.tokenizer(texts, padding=True, truncation=True,
                                max_length=self.cfg["context_tokens"], return_tensors="pt")
        features = self.jepa(images, prepared_pixels=prepared["jepa"]) if prepared is not None else self.jepa(images)
        return encoder(inputs, tokens["input_ids"].to(self.device),
                       tokens["attention_mask"].to(self.device), features, return_image_features=return_image_features)

    def task_inputs(self, images, prepared=None):
        """Image features are always present; slots are optional extra evidence."""
        if self.cfg["slot_conditioning"] or (self.training and hasattr(self, "visual_reconstruction")):
            slots, features = self.encode(images, return_image_features=True, prepared=prepared)
            return self.task_decoder(features, slots if self.cfg["slot_conditioning"] else None), slots
        pixels = self.cfg["vision_pixels"]
        inputs = prepared["vision"] if prepared is not None else self.processor.image_processor(
            images=images, return_tensors="pt", min_pixels=pixels**2, max_pixels=pixels**2)
        inputs = {k: v.to(self.device) for k, v in inputs.items() if torch.is_tensor(v)}
        output = self.encoder.vision(hidden_states=inputs["pixel_values"].to(next(self.encoder.vision.parameters()).dtype),
                                     grid_thw=inputs["image_grid_thw"])
        features = output.pooler_output.reshape(len(images), -1, self.text.output_head.weight.shape[1])
        return self.task_decoder(features), None

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
        # Representation learning only; no temporal classification/report task.
        target = self.encode(**batch["target"], target=True)
        source = self.encode(**batch["source"])
        predicted = self.world(source, batch["delta_hours"].to(self.device))
        latent = F.mse_loss(F.layer_norm(predicted.float(), (1024,)),
                            F.layer_norm(target.float(), (1024,)))
        return self.cfg["latent_weight"] * latent, {"latent": latent.detach()}

    @torch.no_grad()
    def predict(self, task, batch):
        features, slots = self.task_inputs(batch["images"])
        if task == "classification":
            return self.classification(features).sigmoid()
        if task == "segmentation":
            return self.segmentation(features)
        if task == "vqa":
            from .datasets.vqa import INSTRUCTION
            return self.text.generate(features, [INSTRUCTION + q for q in batch["questions"]], self.cfg["generation_tokens"])
        raise ValueError(f"Unknown task: {task}")

    def compact_state(self):
        return {
            "parameters": {n: p.detach().cpu().clone() for n, p in self.named_parameters() if p.requires_grad},
            "pos_weight": self.pos_weight.detach().cpu().clone(),
            "ema": self.target.compact_state(),
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
        if state["ema"] is None:
            raise ValueError("Joint checkpoints require EMA state")
        self.target.restore(state["ema"])
