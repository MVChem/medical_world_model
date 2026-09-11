from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file
from transformers import AutoModel, AutoVideoProcessor

from .backbones import load_letterboxed_rgb, local_hf_revision
from .data import TransitionRecord, load_transition_records
from .io_utils import atomic_write_json, atomic_write_text, jsonl_text, secure_directory

STATE_KEYS = {"source_state", "target_state"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cache only frozen V-JEPA2 current/future states. Qwen remains online "
            "during serious training so its LoRA and latent queries can learn."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vjepa-model", type=Path, required=True)
    parser.add_argument(
        "--split", default="train", choices=["train", "validate", "test"]
    )
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--stack-frames", type=int, default=8)
    parser.add_argument("--single-view-state", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute existing valid state files instead of resuming from them.",
    )
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def distributed_context() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", device_id=device)
    return rank, world_size, local_rank


def atomic_save_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    ) as handle:
        temporary_path = Path(handle.name)
    try:
        save_file(tensors, temporary_path)
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def validate_cached_state(
    path: Path, *, spatial_tokens: int = 256, state_dim: int = 2048
) -> bool:
    """Return whether a state cache is complete and shape-compatible.

    Invalid or interrupted files are recomputed by the extractor. This makes a
    multi-hour extraction safely restartable without trusting file existence.
    """

    try:
        tensors = load_file(path, device="cpu")
    except (OSError, RuntimeError):
        return False
    expected_shape = (spatial_tokens, state_dim)
    return set(tensors) == STATE_KEYS and all(
        tuple(tensors[key].shape) == expected_shape
        and tensors[key].dtype == torch.bfloat16
        for key in STATE_KEYS
    )


class FrozenVJEPAStateExtractor:
    def __init__(
        self,
        *,
        model_path: str | Path,
        device: torch.device,
        stack_frames: int = 8,
        duplicate_view: bool = True,
        dtype: torch.dtype = torch.bfloat16,
    ):
        if stack_frames <= 0 or stack_frames % 2:
            raise ValueError(
                "stack_frames must be a positive multiple of the V-JEPA2 tubelet size (2)"
            )
        self.device = device
        self.dtype = dtype
        self.stack_frames = stack_frames
        self.duplicate_view = duplicate_view
        self.model_source = str(Path(model_path).resolve())
        self.model_revision = local_hf_revision(model_path)
        self.processor = AutoVideoProcessor.from_pretrained(
            model_path, local_files_only=True
        )
        self.model = AutoModel.from_pretrained(
            model_path, dtype=dtype, local_files_only=True
        )
        self.model.requires_grad_(False).eval().to(device)
        if self.model.config.tubelet_size != 2:
            raise ValueError(
                f"expected V-JEPA2 tubelet_size=2, got {self.model.config.tubelet_size}"
            )
        if self.model.config.patch_size != 16 or self.model.config.image_size != 256:
            raise ValueError(
                "the predictor contract expects V-JEPA2 at 256px with 16px patches"
            )

    @torch.inference_mode()
    def encode(self, image_path: Path) -> torch.Tensor:
        image = load_letterboxed_rgb(image_path, 256)
        inputs = self.processor(
            videos=[image] * self.stack_frames,
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
        )
        pixels = inputs["pixel_values_videos"].to(self.device, dtype=self.dtype)
        tokens = self.model.get_vision_features(pixel_values_videos=pixels)
        spatial_tokens = (
            self.model.config.image_size // self.model.config.patch_size
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

    def extract(self, record: TransitionRecord) -> dict[str, torch.Tensor]:
        # Separate calls are required: the video encoder is non-causal.
        source = self.encode(record.source_image)
        target = self.encode(record.target_image)
        if source.shape != target.shape:
            raise ValueError("source and target V-JEPA2 state shapes differ")
        return {"source_state": source, "target_state": target}

    def metadata(self) -> dict[str, Any]:
        output_dim = self.model.config.hidden_size * (2 if self.duplicate_view else 1)
        return {
            "vjepa_model_type": self.model.config.model_type,
            "vjepa_source": self.model_source,
            "vjepa_hidden_size": self.model.config.hidden_size,
            "state_dim": output_dim,
            "vjepa_commit": getattr(self.model.config, "_commit_hash", None)
            or self.model_revision,
            "stack_frames": self.stack_frames,
            "duplicate_view": self.duplicate_view,
            "vjepa_preprocessing": (
                "aspect_preserving_letterbox_256_then_checkpoint_normalization"
            ),
        }


def main() -> None:
    args = parse_args()
    if args.log_every <= 0:
        raise ValueError("--log-every must be positive")
    rank, world_size, local_rank = distributed_context()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("state extraction requires a CUDA GPU")
    torch.cuda.set_device(device)

    records = load_transition_records(
        args.manifest,
        split=args.split,
        max_records=args.max_records,
        require_files=True,
    )
    assigned = records[rank::world_size]
    secure_directory(args.output_dir)
    state_dir = args.output_dir / "states"
    secure_directory(state_dir)

    extractor = FrozenVJEPAStateExtractor(
        model_path=args.vjepa_model,
        device=device,
        stack_frames=args.stack_frames,
        duplicate_view=not args.single_view_state,
    )
    metadata = extractor.metadata()
    expected_dim = int(metadata["state_dim"])

    shard_rows: list[dict[str, object]] = []
    reused = 0
    started = time.monotonic()
    for item_index, record in enumerate(assigned, start=1):
        output_path = state_dir / f"{record.transition_id}.safetensors"
        valid_existing = validate_cached_state(output_path, state_dim=expected_dim)
        if valid_existing and not args.overwrite:
            reused += 1
        else:
            states = extractor.extract(record)
            atomic_save_safetensors(
                output_path,
                {
                    key: value.to(device="cpu", dtype=torch.bfloat16).contiguous()
                    for key, value in states.items()
                },
            )
        shard_rows.append(
            {
                "transition_id": record.transition_id,
                "split": record.split,
                "state_file": str(output_path.relative_to(args.output_dir)),
            }
        )
        if item_index % args.log_every == 0 or item_index == len(assigned):
            elapsed = max(time.monotonic() - started, 1.0e-9)
            print(
                f"rank={rank} states={item_index}/{len(assigned)} "
                f"reused={reused} rate={item_index / elapsed:.3f}/s",
                flush=True,
            )

    shard_path = args.output_dir / f"states.rank-{rank:05d}.jsonl"
    atomic_write_text(shard_path, jsonl_text(shard_rows))
    atomic_write_json(args.output_dir / f"metadata.rank-{rank:05d}.json", metadata)
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        merged: list[dict[str, object]] = []
        for shard_rank in range(world_size):
            path = args.output_dir / f"states.rank-{shard_rank:05d}.jsonl"
            with path.open("r", encoding="utf-8") as handle:
                merged.extend(json.loads(line) for line in handle if line.strip())
        order = {record.transition_id: index for index, record in enumerate(records)}
        merged.sort(key=lambda row: order[str(row["transition_id"])])
        if len(merged) != len(records):
            raise RuntimeError(
                f"merged {len(merged)} rows for {len(records)} source records"
            )
        if len({str(row["transition_id"]) for row in merged}) != len(records):
            raise RuntimeError(
                "merged state manifest contains duplicate transition IDs"
            )
        atomic_write_text(args.output_dir / "states.jsonl", jsonl_text(merged))
        atomic_write_json(
            args.output_dir / "metadata.json",
            {
                "schema_version": "mimic-vla-jepa-states-v1",
                "source_manifest": str(args.manifest.resolve()),
                "split": args.split,
                "records": len(records),
                "world_size": world_size,
                "python": platform.python_version(),
                "torch": torch.__version__,
                "backbone": metadata,
            },
        )

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
