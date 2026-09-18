from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from downstream_tests.common import (
    IXIVolumeDataset,
    attach_ixi_ages,
    autocast_enabled,
    discover_ixi_volumes,
    load_backbone,
    save_json,
    seed_everything,
    split_records,
    split_summary,
)
from downstream_tests.models import AgeRegressor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen V-JEPA probe for IXI age regression"
    )
    parser.add_argument("--config", required=True, help="Pretraining YAML/params YAML")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-key", default="target_encoder")
    parser.add_argument(
        "--volume-root", required=True, help="Root containing IXI NIfTI files"
    )
    parser.add_argument(
        "--metadata", required=True, help="Official IXI.xls or a CSV conversion"
    )
    parser.add_argument("--sequence", default="T1")
    parser.add_argument("--output-dir", default="runs/downstream/ixi_age")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Encoder feature extraction batch"
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-clips", type=int, default=3)
    parser.add_argument("--head-batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=239)
    parser.add_argument("--disable-amp", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def extract_features(
    backbone: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    disable_amp: bool,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    features = []
    ages = []
    subject_ids = []
    for batch_index, batch in enumerate(loader, start=1):
        clips = batch["input"]
        batch_size, num_clips, channels, frames, height, width = clips.shape
        clips = clips.view(-1, channels, frames, height, width).to(
            device, non_blocking=True
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled(device, disable_amp),
        ):
            tokens = backbone(clips)
            pooled = tokens.mean(dim=1)
        pooled = pooled.float().view(batch_size, num_clips, -1).mean(dim=1)
        features.append(pooled.cpu())
        ages.append(batch["age"].float())
        subject_ids.extend(batch["subject_id"])
        if batch_index % 20 == 0 or batch_index == len(loader):
            print(
                f"feature extraction: {batch_index}/{len(loader)} batches", flush=True
            )
    return torch.cat(features), torch.cat(ages), subject_ids


def regression_metrics(
    predictions: torch.Tensor, targets: torch.Tensor
) -> dict[str, float]:
    errors = predictions - targets
    mae = errors.abs().mean().item()
    rmse = math.sqrt(errors.square().mean().item())
    denominator = (targets - targets.mean()).square().sum()
    r2 = 1.0 - errors.square().sum() / denominator if denominator > 0 else torch.nan
    return {"mae_years": mae, "rmse_years": rmse, "r2": float(r2)}


@torch.inference_mode()
def predict(
    head: AgeRegressor,
    features: torch.Tensor,
    age_mean: torch.Tensor,
    age_std: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    return (head(features.to(device)).cpu() * age_std) + age_mean


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = discover_ixi_volumes(args.volume_root, args.sequence)
    records = attach_ixi_ages(records, args.metadata)
    splits = split_records(records, args.seed, args.val_fraction, args.test_fraction)
    print(
        "matched subjects: "
        + ", ".join(f"{name}={len(values)}" for name, values in splits.items())
    )

    backbone, encoder_metadata = load_backbone(
        args.config, args.checkpoint, args.checkpoint_key, device
    )
    extracted = {}
    subject_ids = {}
    for split_name, split_values in splits.items():
        dataset = IXIVolumeDataset(
            split_values,
            frames_per_clip=encoder_metadata["frames_per_clip"],
            crop_size=encoder_metadata["crop_size"],
            num_clips=args.num_clips,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
        )
        features, ages, ids = extract_features(
            backbone, loader, device, args.disable_amp
        )
        extracted[split_name] = (features, ages)
        subject_ids[split_name] = ids
    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()

    train_features, train_ages = extracted["train"]
    val_features, val_ages = extracted["val"]
    test_features, test_ages = extracted["test"]
    age_mean = train_ages.mean()
    age_std = train_ages.std().clamp_min(1e-6)
    normalized_train_ages = (train_ages - age_mean) / age_std

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        TensorDataset(train_features, normalized_train_ages),
        batch_size=args.head_batch_size,
        shuffle=True,
        generator=generator,
    )
    head = AgeRegressor(encoder_metadata["embed_dim"], args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    best_val_mae = float("inf")
    best_epoch = 0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        head.train()
        for features, targets in train_loader:
            predictions = head(features.to(device))
            loss = F.smooth_l1_loss(predictions, targets.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        head.eval()
        val_predictions = predict(head, val_features, age_mean, age_std, device)
        val_mae = (val_predictions - val_ages).abs().mean().item()
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            best_state = copy.deepcopy(head.state_dict())
        if epoch == 1 or epoch % 20 == 0 or epoch == args.epochs:
            print(f"epoch={epoch:03d} val_mae={val_mae:.3f} years", flush=True)

    if best_state is None:
        raise RuntimeError("No age regressor checkpoint was selected")
    head.load_state_dict(best_state)
    head.eval()
    test_predictions = predict(head, test_features, age_mean, age_std, device)
    test_metrics = regression_metrics(test_predictions, test_ages)
    baseline_predictions = torch.full_like(test_ages, age_mean)
    baseline_metrics = regression_metrics(baseline_predictions, test_ages)

    torch.save(
        {
            "regressor": head.cpu().state_dict(),
            "embed_dim": encoder_metadata["embed_dim"],
            "hidden_dim": args.hidden_dim,
            "age_mean": float(age_mean),
            "age_std": float(age_std),
            "best_epoch": best_epoch,
        },
        output_dir / "best_regressor.pt",
    )
    pd.DataFrame(
        {
            "subject_id": subject_ids["test"],
            "age": test_ages.numpy(),
            "predicted_age": test_predictions.numpy(),
            "absolute_error": (test_predictions - test_ages).abs().numpy(),
        }
    ).to_csv(output_dir / "test_predictions.csv", index=False)

    payload = {
        "task": "ixi_age_regression",
        "encoder": encoder_metadata,
        "sequence": args.sequence.upper(),
        "seed": args.seed,
        "num_clips": args.num_clips,
        "best_epoch": best_epoch,
        "best_val_mae_years": best_val_mae,
        "test": test_metrics,
        "train_mean_baseline": baseline_metrics,
        "splits": split_summary(splits),
    }
    save_json(payload, output_dir / "metrics.json")
    print(f"test metrics: {test_metrics}")
    print(f"train-mean baseline: {baseline_metrics}")
    print(f"saved results to {output_dir}")


if __name__ == "__main__":
    main()
