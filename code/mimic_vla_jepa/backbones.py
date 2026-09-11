from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps
from transformers import (
    AutoModel,
    AutoProcessor,
    AutoVideoProcessor,
    Qwen3_5ForConditionalGeneration,
)

from .data import TransitionRecord

HF_REVISION = re.compile(r"^[0-9a-f]{40}$")


def local_hf_revision(model_path: str | Path) -> str | None:
    """Read the resolved Hub commit recorded by huggingface_hub.

    Transformers does not populate ``config._commit_hash`` when a snapshot was
    downloaded into an ordinary local directory. The adjacent download
    metadata retains that revision, so include it in restricted feature
    manifests for reproducibility.
    """

    metadata_path = (
        Path(model_path)
        / ".cache"
        / "huggingface"
        / "download"
        / "config.json.metadata"
    )
    try:
        revision = metadata_path.read_text(encoding="utf-8").splitlines()[0].strip()
    except (OSError, IndexError):
        return None
    return revision if HF_REVISION.fullmatch(revision) else None


@dataclass(frozen=True)
class ExtractedFeatures:
    source_state: torch.Tensor
    target_state: torch.Tensor
    query_state: torch.Tensor


def query_positions(
    input_ids: torch.Tensor, token_id: int, expected: int
) -> torch.Tensor:
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(
            f"feature extraction expects input_ids [1, L], got {tuple(input_ids.shape)}"
        )
    positions = torch.nonzero(input_ids[0] == token_id, as_tuple=False).flatten()
    if positions.numel() != expected:
        raise ValueError(
            f"expected {expected} latent query positions, found {positions.numel()}"
        )
    return positions


def load_letterboxed_rgb(image_path: Path, size: int) -> Image.Image:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB").copy()
    contained = ImageOps.contain(rgb, (size, size), method=Image.Resampling.BICUBIC)
    letterboxed = Image.new("RGB", (size, size), color=(0, 0, 0))
    offset = ((size - contained.width) // 2, (size - contained.height) // 2)
    letterboxed.paste(contained, offset)
    return letterboxed


class FrozenBackboneExtractor:
    def __init__(
        self,
        *,
        qwen_model: str | Path,
        vjepa_model: str | Path,
        device: torch.device,
        query_token: str = "<|latent_0|>",
        query_tokens: int = 24,
        stack_frames: int = 8,
        duplicate_view: bool = True,
        dtype: torch.dtype = torch.bfloat16,
    ):
        if query_tokens <= 0:
            raise ValueError("query_tokens must be positive")
        if stack_frames <= 0 or stack_frames % 2:
            raise ValueError(
                "stack_frames must be a positive multiple of the V-JEPA2 tubelet size (2)"
            )
        self.device = device
        self.dtype = dtype
        self.query_token = query_token
        self.query_tokens = query_tokens
        self.stack_frames = stack_frames
        self.duplicate_view = duplicate_view
        self.qwen_source = str(Path(qwen_model).resolve())
        self.vjepa_source = str(Path(vjepa_model).resolve())
        self.qwen_revision = local_hf_revision(qwen_model)
        self.vjepa_revision = local_hf_revision(vjepa_model)

        self.qwen_processor = AutoProcessor.from_pretrained(
            qwen_model, local_files_only=True
        )
        self.qwen_processor.tokenizer.padding_side = "left"
        added = self.qwen_processor.tokenizer.add_special_tokens(
            {"additional_special_tokens": [self.query_token]}
        )
        self.query_token_id = self.qwen_processor.tokenizer.convert_tokens_to_ids(
            self.query_token
        )
        if self.query_token_id == self.qwen_processor.tokenizer.unk_token_id:
            raise RuntimeError("failed to add the latent query token")

        self.qwen = Qwen3_5ForConditionalGeneration.from_pretrained(
            qwen_model,
            dtype=dtype,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        if added:
            self.qwen.resize_token_embeddings(
                len(self.qwen_processor.tokenizer), mean_resizing=False
            )
            # Make independently extracted feature shards reproducible across
            # ranks. The special token is a query marker, so a deterministic
            # EOS initialization is preferable to per-process random rows.
            with torch.no_grad():
                embeddings = self.qwen.get_input_embeddings().weight
                embeddings[self.query_token_id].copy_(
                    embeddings[self.qwen_processor.tokenizer.eos_token_id]
                )
        self.qwen.requires_grad_(False).eval().to(device)

        self.vjepa_processor = AutoVideoProcessor.from_pretrained(
            vjepa_model, local_files_only=True
        )
        self.vjepa = AutoModel.from_pretrained(
            vjepa_model,
            dtype=dtype,
            local_files_only=True,
        )
        self.vjepa.requires_grad_(False).eval().to(device)
        if self.vjepa.config.tubelet_size != 2:
            raise ValueError(
                f"expected V-JEPA2 tubelet_size=2, got {self.vjepa.config.tubelet_size}"
            )
        if self.vjepa.config.patch_size != 16 or self.vjepa.config.image_size != 256:
            raise ValueError(
                "the released predictor contract expects V-JEPA2 at 256px with 16px patches"
            )

    def _qwen_inputs(self, image_path: Path, prompt: str) -> dict[str, torch.Tensor]:
        query_suffix = "".join([self.query_token] * self.query_tokens)
        image = load_letterboxed_rgb(image_path, 224)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": f"{prompt}\nLATENT TRANSITION QUERIES:\n{query_suffix}",
                    },
                ],
            }
        ]
        inputs = self.qwen_processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={
                "text_kwargs": {
                    "padding": True,
                    "truncation": True,
                    "max_length": 2048,
                },
                "images_kwargs": {
                    "min_pixels": 224 * 224,
                    "max_pixels": 224 * 224,
                },
            },
        )
        expected_grid = torch.tensor([[1, 14, 14]])
        if "image_grid_thw" not in inputs or not torch.equal(
            inputs["image_grid_thw"].cpu(), expected_grid
        ):
            actual = inputs.get("image_grid_thw")
            raise ValueError(f"expected Qwen image grid [[1, 14, 14]], got {actual}")
        return {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }

    @torch.inference_mode()
    def encode_query(self, image_path: Path, prompt: str) -> torch.Tensor:
        inputs = self._qwen_inputs(image_path, prompt)
        positions = query_positions(
            inputs["input_ids"], self.query_token_id, self.query_tokens
        )
        # Call the multimodal backbone directly. The conditional-generation
        # wrapper would materialize logits over the ~248K-token vocabulary even
        # though this baseline only needs hidden states at the query positions.
        outputs = self.qwen.model(
            **inputs,
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        return hidden[0, positions, :].detach()

    def _video_inputs(self, image_path: Path) -> torch.Tensor:
        letterboxed = load_letterboxed_rgb(image_path, 256)
        frames = [letterboxed] * self.stack_frames
        inputs = self.vjepa_processor(
            videos=frames,
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
        )
        return inputs["pixel_values_videos"].to(self.device, dtype=self.dtype)

    @torch.inference_mode()
    def encode_state(self, image_path: Path) -> torch.Tensor:
        pixel_values = self._video_inputs(image_path)
        tokens = self.vjepa.get_vision_features(pixel_values_videos=pixel_values)
        spatial_tokens = (
            self.vjepa.config.image_size // self.vjepa.config.patch_size
        ) ** 2
        if tokens.shape[0] != 1 or tokens.shape[1] % spatial_tokens:
            raise ValueError(f"unexpected V-JEPA2 token shape: {tuple(tokens.shape)}")
        temporal_tokens = tokens.shape[1] // spatial_tokens
        state = tokens.reshape(
            1, temporal_tokens, spatial_tokens, tokens.shape[-1]
        ).mean(dim=1)[0]
        if self.duplicate_view:
            state = torch.cat([state, state], dim=-1)
        return state.detach()

    @torch.inference_mode()
    def extract(self, record: TransitionRecord) -> ExtractedFeatures:
        # This call boundary is deliberate: Qwen receives the source path and current-only prompt;
        # the target path is used only by the stopped-gradient target encoder.
        query_state = self.encode_query(record.source_image, record.prompt)
        source_state = self.encode_state(record.source_image)
        target_state = self.encode_state(record.target_image)
        if source_state.shape != target_state.shape:
            raise ValueError("source and target state shapes differ")
        return ExtractedFeatures(
            source_state=source_state,
            target_state=target_state,
            query_state=query_state,
        )

    def model_metadata(self) -> dict[str, Any]:
        return {
            "qwen_model_type": self.qwen.config.model_type,
            "qwen_source": self.qwen_source,
            "qwen_hidden_size": self.qwen.config.text_config.hidden_size,
            "qwen_commit": getattr(self.qwen.config, "_commit_hash", None)
            or self.qwen_revision,
            "vjepa_model_type": self.vjepa.config.model_type,
            "vjepa_source": self.vjepa_source,
            "vjepa_hidden_size": self.vjepa.config.hidden_size,
            "vjepa_commit": getattr(self.vjepa.config, "_commit_hash", None)
            or self.vjepa_revision,
            "query_token": self.query_token,
            "query_tokens": self.query_tokens,
            "stack_frames": self.stack_frames,
            "duplicate_view": self.duplicate_view,
            "vjepa_preprocessing": "aspect_preserving_letterbox_256_then_checkpoint_normalization",
        }
