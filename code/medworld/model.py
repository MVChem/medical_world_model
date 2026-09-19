"""Unified current-task and signed-time model. No observation bypass in text decoding."""
import torch
from torch import nn
import torch.nn.functional as F

from .adaptation import LANGUAGE_TARGETS, adapt_selected, depths, shared_frozen_copy
from .downstream_tasks.classification import ClassificationHead, finding_loss
from .downstream_tasks.report import ReportDecoder
from .downstream_tasks.segmentation.decoder import SegmentationHead
from .downstream_tasks.super_resolution.decoder import SuperResolutionHead
from .downstream_tasks.training import current_loss as current_task_loss
from .ema import EMATarget
from .encoder import FrozenJEPA, StateEncoder
from .predictor import WorldModel

TEMPORAL_COLUMNS = (0, 1, 2, 3, 8, 11)


class MedWorld(nn.Module):
    def __init__(self, cfg, device="cuda"):
        super().__init__()
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
        self.cfg = dict(cfg)
        base = Qwen3_5ForConditionalGeneration.from_pretrained(
            cfg["qwen"], local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa")
        base.requires_grad_(False)
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
        self.report = ReportDecoder(decoder_language, base.lm_head, self.tokenizer, cfg)
        self.jepa = FrozenJEPA(cfg["jepa"])
        self.world = WorldModel(cfg)
        self.classification = ClassificationHead()
        self.segmentation, self.sr = SegmentationHead(), SuperResolutionHead()
        if cfg.get("visual_consistency_weight", 0) > 0:
            from .representation import SlotSpatialReconstruction
            # Do not perturb decoder initialization or the training RNG stream.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(cfg["seed"] + 9138)
                self.visual_reconstruction = SlotSpatialReconstruction()
        self.register_buffer("pos_weight", torch.ones(13))
        self.target = None
        self.metadata = {
            "representation_objective": "visual_slot_spatial_consistency" if hasattr(self, "visual_reconstruction") else None,
            "architecture": "medworld_unified_4plus4_v1", "state_shape": [8, 1024],
            "language_depths_1based": [i + 1 for i in depths(len(self.encoder.language.layers))],
            "vision_depths_1based": [i + 1 for i in depths(len(self.encoder.vision.blocks))],
            "target_update": "EMA of all trainable state encoder parameters after each Stage 2 optimizer update",
            "text_decoder_inputs": ["state", "fixed task prompt"],
            "time_condition": "signed realized_gap_hours; direction is a predictor input only",
            "current_tasks": ["classification", "report", "segmentation", "sr"],
            "temporal_inputs": ["source image", "source report", "signed delta_hours"],
            "report_only_input": False, "ehr_input": False,
            "frozen_backbone_shared": True, "encoder_decoder_lora_shared": False,
        }
        self.to(device)

    @property
    def device(self):
        return self.encoder.slot_queries.device

    def begin_stage2(self):
        if self.target is not None:
            raise ValueError("EMA already initialized; restoring must preserve its saved state")
        self.target = EMATarget(self.encoder).to(self.device)

    def forward(self, task, batch, replay_task=None, replay_batch=None):
        """One DDP forward owns all graphs for an optimizer microbatch."""
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                            enabled=self.cfg.get("amp", False)):
            loss, parts = self.temporal_loss(batch) if task == "temporal" else self.current_loss(task, batch)
            if replay_task is not None:
                if task != "temporal" or replay_batch is None:
                    raise ValueError("Replay requires a temporal training batch")
                replay, replay_parts = self.current_loss(replay_task, replay_batch)
                loss = loss + self.cfg["replay_weight"] * replay
                parts.update({"replay_" + key: value for key, value in replay_parts.items()})
        return loss, parts

    def encode(self, images, texts=None, *, spatial=False, target=False):
        if not images:
            raise ValueError("At least one observation image is required")
        encoder = self.target if target else self.encoder
        if encoder is None:
            raise ValueError("Initialize Stage 2 before using the target encoder")
        pixels = self.cfg["vision_pixels"]
        inputs = self.processor.image_processor(images=images, return_tensors="pt",
                                                min_pixels=pixels**2, max_pixels=pixels**2)
        inputs = {k: v.to(self.device) for k, v in inputs.items() if torch.is_tensor(v)}
        if spatial:
            return encoder(inputs, spatial=True)
        if texts is None:
            texts = [""] * len(images)
        if len(texts) != len(images) or any(not isinstance(t, str) for t in texts):
            raise ValueError("Expected one observation text per image")
        # Identical instruction at every time point. The encoder never sees the
        # requested direction, time gap, or the other observation's evidence.
        texts = ["Chest radiograph observation." + ("\nReport:\n" + t if t else "") for t in texts]
        tokens = self.tokenizer(texts, padding=True, truncation=True,
                                max_length=self.cfg["context_tokens"], return_tensors="pt")
        features = self.jepa(images)
        return encoder(inputs, tokens["input_ids"].to(self.device),
                       tokens["attention_mask"].to(self.device), features)

    def current_loss(self, task, batch):
        loss, parts, state = current_task_loss(self, task, batch, return_state=True)
        if (self.training and torch.is_grad_enabled() and task in ("segmentation", "sr")
                and hasattr(self, "visual_reconstruction")):
            auxiliary = self.visual_reconstruction.loss(
                state, batch["pixels"].to(self.device), batch["mask"].to(self.device),
                batch["ids"], self.visual_teacher, self.processor, self.cfg)
            weighted = self.cfg["visual_consistency_weight"] * auxiliary
            parts.update(visual_consistency=auxiliary.detach(), visual_consistency_weighted=weighted.detach())
            loss = loss + weighted
        return loss, parts

    def temporal_loss(self, batch, *, audit=False):
        if self.target is None:
            raise ValueError("Stage 2 requires an EMA target")
        target = self.encode(**batch["target"], target=True)
        source = self.encode(**batch["source"])
        predicted = self.world(source, batch["delta_hours"].to(self.device))
        latent = F.mse_loss(F.layer_norm(predicted.float(), (1024,)),
                            F.layer_norm(target.float(), (1024,)))
        report = self.report.loss(predicted, batch["report_targets"])
        finding = finding_loss(self.classification(predicted)[:, TEMPORAL_COLUMNS], batch["labels"])
        loss = (self.cfg["latent_weight"] * latent + self.cfg["report_weight"] * report
                + self.cfg["finding_weight"] * finding)
        parts = {"latent": latent.detach(), "report": report.detach(), "finding": finding.detach()}
        if audit:
            # Expose differentiable scalars only on explicit request, never store
            # activation graphs on the module (which is later copied for EMA).
            return loss, parts, {"report": report, "source": source, "predicted": predicted, "target": target}
        return loss, parts

    @torch.no_grad()
    def predict_state(self, source, delta_hours):
        return self.world(self.encode(**source), delta_hours.to(self.device))

    @torch.no_grad()
    def decode_state(self, state, max_new_tokens=None):
        return self.report.generate(state, self.cfg["generation_tokens"] if max_new_tokens is None else max_new_tokens)

    def compact_state(self):
        return {
            "parameters": {n: p.detach().cpu().clone() for n, p in self.named_parameters() if p.requires_grad},
            "pos_weight": self.pos_weight.detach().cpu().clone(),
            "ema": None if self.target is None else self.target.compact_state(),
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
        if state["ema"] is not None:
            if self.target is None:
                self.begin_stage2()
            self.target.restore(state["ema"])
        elif self.target is not None:
            raise ValueError("Cannot discard an initialized EMA target")
