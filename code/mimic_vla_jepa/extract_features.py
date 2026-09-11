from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import save_file

from .backbones import FrozenBackboneExtractor
from .data import load_transition_records
from .io_utils import atomic_write_json, atomic_write_text, jsonl_text, secure_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frozen Qwen3.5 and V-JEPA2 features from MIMIC pairs"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qwen-model", type=Path, required=True)
    parser.add_argument("--vjepa-model", type=Path, required=True)
    parser.add_argument(
        "--split", default="train", choices=["train", "validate", "test"]
    )
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--query-tokens", type=int, default=24)
    parser.add_argument("--stack-frames", type=int, default=8)
    parser.add_argument("--single-view-state", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
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
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = distributed_context()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("feature extraction requires a CUDA GPU")
    torch.cuda.set_device(device)

    records = load_transition_records(
        args.manifest,
        split=args.split,
        max_records=args.max_records,
        require_files=True,
    )
    assigned = records[rank::world_size]
    secure_directory(args.output_dir)
    feature_dir = args.output_dir / "features"
    secure_directory(feature_dir)

    extractor = None
    metadata = None
    if assigned:
        extractor = FrozenBackboneExtractor(
            qwen_model=args.qwen_model,
            vjepa_model=args.vjepa_model,
            device=device,
            query_tokens=args.query_tokens,
            stack_frames=args.stack_frames,
            duplicate_view=not args.single_view_state,
        )
        metadata = extractor.model_metadata()

    shard_rows: list[dict[str, object]] = []
    for item_index, record in enumerate(assigned, start=1):
        output_path = feature_dir / f"{record.transition_id}.safetensors"
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"refusing to replace existing feature file: {output_path}"
            )
        assert extractor is not None
        features = extractor.extract(record)
        tensors = {
            "source_state": features.source_state.to(
                device="cpu", dtype=torch.bfloat16
            ).contiguous(),
            "target_state": features.target_state.to(
                device="cpu", dtype=torch.bfloat16
            ).contiguous(),
            "query_state": features.query_state.to(
                device="cpu", dtype=torch.bfloat16
            ).contiguous(),
        }
        atomic_save_safetensors(output_path, tensors)
        shard_rows.append(
            {
                "transition_id": record.transition_id,
                "split": record.split,
                "feature_file": str(output_path.relative_to(args.output_dir)),
            }
        )
        print(
            f"rank={rank} extracted={item_index}/{len(assigned)} id={record.transition_id}",
            flush=True,
        )

    shard_path = args.output_dir / f"features.rank-{rank:05d}.jsonl"
    atomic_write_text(shard_path, jsonl_text(shard_rows))
    if metadata is not None:
        atomic_write_json(args.output_dir / f"metadata.rank-{rank:05d}.json", metadata)
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        merged: list[dict[str, object]] = []
        for shard_rank in range(world_size):
            path = args.output_dir / f"features.rank-{shard_rank:05d}.jsonl"
            with path.open("r", encoding="utf-8") as handle:
                merged.extend(json.loads(line) for line in handle if line.strip())
        order = {record.transition_id: index for index, record in enumerate(records)}
        merged.sort(key=lambda row: order[str(row["transition_id"])])
        if len(merged) != len(records):
            raise RuntimeError(
                f"merged {len(merged)} rows for {len(records)} source records"
            )
        atomic_write_text(args.output_dir / "features.jsonl", jsonl_text(merged))
        metadata_value = metadata
        if metadata_value is None:
            for shard_rank in range(world_size):
                candidate = args.output_dir / f"metadata.rank-{shard_rank:05d}.json"
                if candidate.exists():
                    metadata_value = json.loads(candidate.read_text(encoding="utf-8"))
                    break
        atomic_write_json(
            args.output_dir / "metadata.json",
            {
                "schema_version": "mimic-vla-jepa-features-v1",
                "source_manifest": str(args.manifest.resolve()),
                "split": args.split,
                "records": len(records),
                "world_size": world_size,
                "python": platform.python_version(),
                "torch": torch.__version__,
                "backbones": metadata_value,
            },
        )

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
