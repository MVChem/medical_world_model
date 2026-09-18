from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

from downstream_tests.common import (
    IXIVolumeDataset,
    autocast_enabled,
    discover_ixi_volumes,
    load_backbone,
    save_json,
    seed_everything,
    split_records,
    split_summary,
)
from downstream_tests.models import PatchDecoder3D


class CachedReconstructionDataset(Dataset):
    def __init__(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor,
        subject_ids: list[str],
    ) -> None:
        self.tokens = tokens
        self.targets = targets
        self.subject_ids = subject_ids

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, index: int) -> dict:
        return {
            "tokens": self.tokens[index],
            "target": self.targets[index],
            "subject_id": self.subject_ids[index],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen V-JEPA token-to-pixel reconstruction probe"
    )
    parser.add_argument("--config", required=True, help="Pretraining YAML/params YAML")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-key", default="target_encoder")
    parser.add_argument(
        "--volume-root", required=True, help="Root containing IXI NIfTI files"
    )
    parser.add_argument("--sequence", default="T1")
    parser.add_argument("--output-dir", default="runs/downstream/ixi_reconstruction")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--foreground-threshold", type=float, default=0.05)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=239)
    parser.add_argument("--visualization-subjects", type=int, default=4)
    parser.add_argument("--disable-amp", action="store_true")
    return parser.parse_args()


def foreground_aware_loss(
    reconstruction: torch.Tensor, target: torch.Tensor, threshold: float
) -> torch.Tensor:
    squared_error = (reconstruction - target).square()
    foreground = target.mean(dim=1, keepdim=True) > threshold
    foreground = foreground.expand_as(squared_error)
    foreground_mse = (
        squared_error[foreground].mean() if foreground.any() else squared_error.mean()
    )
    return squared_error.mean() + foreground_mse


def update_sums(
    sums: dict[str, float],
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
) -> None:
    squared_error = (reconstruction.float() - target.float()).square()
    foreground = target.float().mean(dim=1, keepdim=True) > threshold
    foreground = foreground.expand_as(squared_error)
    sums["squared_error"] += squared_error.sum().item()
    sums["values"] += squared_error.numel()
    sums["foreground_squared_error"] += squared_error[foreground].sum().item()
    sums["foreground_values"] += foreground.sum().item()


def finalize_metrics(sums: dict[str, float]) -> dict[str, float]:
    mse = sums["squared_error"] / max(1.0, sums["values"])
    foreground_mse = sums["foreground_squared_error"] / max(
        1.0, sums["foreground_values"]
    )
    return {
        "mse": mse,
        "psnr_db": -10.0 * math.log10(max(mse, 1e-12)),
        "foreground_mse": foreground_mse,
        "foreground_psnr_db": -10.0 * math.log10(max(foreground_mse, 1e-12)),
    }


@torch.inference_mode()
def extract_tokens(
    backbone: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    disable_amp: bool,
) -> CachedReconstructionDataset:
    token_batches = []
    target_batches = []
    subject_ids = []
    for batch_index, batch in enumerate(loader, start=1):
        clips = batch["input"][:, 0].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled(device, disable_amp),
        ):
            tokens = backbone(clips)
        token_batches.append(tokens.to(dtype=torch.bfloat16).cpu())
        targets = batch["target"][:, 0].to(torch.float16)
        target_batches.append(targets)
        subject_ids.extend(batch["subject_id"])
        if batch_index % 10 == 0 or batch_index == len(loader):
            print(f"token extraction: {batch_index}/{len(loader)} batches", flush=True)
    return CachedReconstructionDataset(
        torch.cat(token_batches), torch.cat(target_batches), subject_ids
    )


@torch.inference_mode()
def evaluate(
    decoder: PatchDecoder3D,
    loader: DataLoader,
    device: torch.device,
    frames: int,
    image_size: int,
    threshold: float,
    disable_amp: bool,
    num_examples: int = 0,
) -> tuple[dict[str, float], list[tuple[str, torch.Tensor, torch.Tensor]]]:
    decoder.eval()
    sums = {
        "squared_error": 0.0,
        "values": 0.0,
        "foreground_squared_error": 0.0,
        "foreground_values": 0.0,
    }
    examples = []
    for batch in loader:
        tokens = batch["tokens"].to(device, non_blocking=True)
        if not autocast_enabled(device, disable_amp):
            tokens = tokens.float()
        target = batch["target"].to(device, non_blocking=True).float()
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled(device, disable_amp),
        ):
            reconstruction = decoder(tokens, frames, image_size, image_size)
        update_sums(sums, reconstruction, target, threshold)
        remaining = num_examples - len(examples)
        for index in range(min(remaining, target.shape[0])):
            examples.append(
                (
                    batch["subject_id"][index],
                    target[index].cpu(),
                    reconstruction[index].float().cpu(),
                )
            )
    return finalize_metrics(sums), examples


def _grayscale_image(frame: torch.Tensor) -> Image.Image:
    pixels = (frame.squeeze().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
    return Image.fromarray(pixels.numpy(), mode="L").convert("RGB")


def _error_image(target: torch.Tensor, reconstruction: torch.Tensor) -> Image.Image:
    error = (target - reconstruction).abs().squeeze()
    red = (error * 4.0).clamp(0.0, 1.0)
    green = (error * 8.0 - 1.0).clamp(0.0, 1.0)
    blue = torch.zeros_like(error)
    heatmap = torch.stack((red, green, blue), dim=-1)
    pixels = (heatmap * 255.0).round().to(torch.uint8).numpy()
    return Image.fromarray(pixels, mode="RGB")


def save_reconstruction_examples(
    examples: list[tuple[str, torch.Tensor, torch.Tensor]], path: Path
) -> None:
    if not examples:
        raise ValueError("At least one reconstruction example is required")
    image_size = examples[0][1].shape[-1]
    header_height = 30
    label_height = 20
    rows = []
    for subject_id, target, reconstruction in examples:
        frames = target.shape[1]
        selected = sorted({frames // 4, frames // 2, (3 * frames) // 4})
        rows.extend((subject_id, frame, target, reconstruction) for frame in selected)

    canvas = Image.new(
        "RGB",
        (3 * image_size, header_height + len(rows) * (label_height + image_size)),
        color="white",
    )
    draw = ImageDraw.Draw(canvas)
    for column, title in enumerate(
        ("Target", "Reconstruction", "|Error| (4x color scale)")
    ):
        draw.text((column * image_size + 8, 8), title, fill="black")

    for row_index, (subject_id, frame, target, reconstruction) in enumerate(rows):
        top = header_height + row_index * (label_height + image_size)
        draw.text((8, top + 3), f"{subject_id}  slice {frame}", fill="black")
        image_top = top + label_height
        canvas.paste(_grayscale_image(target[:, frame]), (0, image_top))
        canvas.paste(
            _grayscale_image(reconstruction[:, frame]), (image_size, image_top)
        )
        canvas.paste(
            _error_image(target[:, frame], reconstruction[:, frame]),
            (2 * image_size, image_top),
        )
    canvas.save(path)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = discover_ixi_volumes(args.volume_root, args.sequence)
    splits = split_records(records, args.seed, args.val_fraction, args.test_fraction)
    print(
        "subjects: "
        + ", ".join(f"{name}={len(values)}" for name, values in splits.items())
    )
    backbone, encoder_metadata = load_backbone(
        args.config, args.checkpoint, args.checkpoint_key, device
    )

    feature_datasets = {}
    for split_name, split_values in splits.items():
        dataset = IXIVolumeDataset(
            split_values,
            frames_per_clip=encoder_metadata["frames_per_clip"],
            crop_size=encoder_metadata["crop_size"],
            num_clips=1,
            return_target=True,
        )
        extraction_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
        )
        feature_datasets[split_name] = extract_tokens(
            backbone, extraction_loader, device, args.disable_amp
        )
    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()

    loaders = {}
    generator = torch.Generator().manual_seed(args.seed)
    for split_name, dataset in feature_datasets.items():
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=split_name == "train",
            generator=generator if split_name == "train" else None,
            pin_memory=device.type == "cuda",
        )

    decoder = PatchDecoder3D(
        embed_dim=encoder_metadata["embed_dim"],
        patch_size=encoder_metadata["patch_size"],
        tubelet_size=encoder_metadata["tubelet_size"],
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        decoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    best_val_foreground_mse = float("inf")
    best_epoch = 0
    best_state = None
    frames = encoder_metadata["frames_per_clip"]
    image_size = encoder_metadata["crop_size"]
    for epoch in range(1, args.epochs + 1):
        decoder.train()
        loss_sum = 0.0
        for batch in loaders["train"]:
            tokens = batch["tokens"].to(device, non_blocking=True)
            if not autocast_enabled(device, args.disable_amp):
                tokens = tokens.float()
            target = batch["target"].to(device, non_blocking=True).float()
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=autocast_enabled(device, args.disable_amp),
            ):
                reconstruction = decoder(tokens, frames, image_size, image_size)
                loss = foreground_aware_loss(
                    reconstruction, target, args.foreground_threshold
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        val_metrics, _ = evaluate(
            decoder,
            loaders["val"],
            device,
            frames,
            image_size,
            args.foreground_threshold,
            args.disable_amp,
        )
        if val_metrics["foreground_mse"] < best_val_foreground_mse:
            best_val_foreground_mse = val_metrics["foreground_mse"]
            best_epoch = epoch
            best_state = copy.deepcopy(decoder.state_dict())
        print(
            f"epoch={epoch:03d} train_loss={loss_sum / len(loaders['train']):.6f} "
            f"val_psnr={val_metrics['psnr_db']:.2f} "
            f"val_foreground_psnr={val_metrics['foreground_psnr_db']:.2f}",
            flush=True,
        )

    if best_state is None:
        raise RuntimeError("No reconstruction decoder checkpoint was selected")
    decoder.load_state_dict(best_state)
    test_metrics, examples = evaluate(
        decoder,
        loaders["test"],
        device,
        frames,
        image_size,
        args.foreground_threshold,
        args.disable_amp,
        num_examples=args.visualization_subjects,
    )

    torch.save(
        {
            "decoder": decoder.cpu().state_dict(),
            "embed_dim": encoder_metadata["embed_dim"],
            "hidden_dim": args.hidden_dim,
            "patch_size": encoder_metadata["patch_size"],
            "tubelet_size": encoder_metadata["tubelet_size"],
            "best_epoch": best_epoch,
        },
        output_dir / "best_decoder.pt",
    )
    save_reconstruction_examples(examples, output_dir / "reconstructions.png")
    payload = {
        "task": "ixi_reconstruction",
        "encoder": encoder_metadata,
        "sequence": args.sequence.upper(),
        "seed": args.seed,
        "foreground_threshold": args.foreground_threshold,
        "best_epoch": best_epoch,
        "best_val_foreground_mse": best_val_foreground_mse,
        "test": test_metrics,
        "splits": split_summary(splits),
    }
    save_json(payload, output_dir / "metrics.json")
    print(f"test metrics: {test_metrics}")
    print(f"saved results to {output_dir}")


if __name__ == "__main__":
    main()
