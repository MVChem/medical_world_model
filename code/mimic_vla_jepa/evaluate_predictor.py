from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .features import CachedFeatureDataset
from .io_utils import atomic_write_json
from .predictor import PredictorConfig, VLAJEPAPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a cached-feature predictor against copy/query ablations"
    )
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def distances(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    per_example_l1 = (prediction.float() - target.float()).abs().flatten(1).mean(dim=1)
    per_example_cosine = 1.0 - F.cosine_similarity(
        prediction.float().flatten(1), target.float().flatten(1), dim=-1
    )
    return {
        "l1": float(per_example_l1.mean().item()),
        "cosine_distance": float(per_example_cosine.mean().item()),
    }


def per_example_l1(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction.float() - target.float()).abs().flatten(1).mean(dim=1)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("evaluation requires a CUDA GPU")
    device = torch.device("cuda:0")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = PredictorConfig(**checkpoint["predictor_config"])
    model = VLAJEPAPredictor(config).to(device).eval()
    model.load_state_dict(checkpoint["model"])

    dataset = CachedFeatureDataset(args.features, split=args.split)
    if len(dataset) < 2:
        raise ValueError("query shuffling requires at least two evaluation examples")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    totals: dict[str, list[tuple[int, dict[str, float]]]] = {
        "prediction": [],
        "copy_state": [],
        "zero_query": [],
        "shuffled_query": [],
    }
    paired_query_totals = {
        "shuffled_minus_correct_l1": 0.0,
        "zero_minus_correct_l1": 0.0,
        "prediction_change_l1_shuffled": 0.0,
        "prediction_change_l1_zero": 0.0,
    }
    examples = 0
    with torch.inference_mode():
        offset = 0
        for batch in loader:
            source = batch["source_state"].to(device, dtype=torch.float32)
            target = batch["target_state"].to(device, dtype=torch.float32)
            query = batch["query_state"].to(device, dtype=torch.float32)
            # Rotate queries over the complete dataset, rather than only within
            # each batch. This remains a true shuffle for a final batch of one.
            shuffled_indices = [
                (index - 1) % len(dataset)
                for index in range(offset, offset + source.shape[0])
            ]
            shuffled_query = torch.stack(
                [dataset[index]["query_state"] for index in shuffled_indices]
            ).to(device, dtype=torch.float32)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prediction = model(source, query)
                zero_prediction = model(source, torch.zeros_like(query))
                shuffled_prediction = model(source, shuffled_query)
            batch_examples = source.shape[0]
            totals["prediction"].append((batch_examples, distances(prediction, target)))
            totals["copy_state"].append((batch_examples, distances(source, target)))
            totals["zero_query"].append(
                (batch_examples, distances(zero_prediction, target))
            )
            totals["shuffled_query"].append(
                (batch_examples, distances(shuffled_prediction, target))
            )
            correct_l1 = per_example_l1(prediction, target)
            shuffled_l1 = per_example_l1(shuffled_prediction, target)
            zero_l1 = per_example_l1(zero_prediction, target)
            paired_query_totals["shuffled_minus_correct_l1"] += float(
                (shuffled_l1 - correct_l1).sum().item()
            )
            paired_query_totals["zero_minus_correct_l1"] += float(
                (zero_l1 - correct_l1).sum().item()
            )
            paired_query_totals["prediction_change_l1_shuffled"] += float(
                per_example_l1(shuffled_prediction, prediction).sum().item()
            )
            paired_query_totals["prediction_change_l1_zero"] += float(
                per_example_l1(zero_prediction, prediction).sum().item()
            )
            examples += batch_examples
            offset += batch_examples

    metrics = {
        name: {
            key: sum(batch_examples * item[key] for batch_examples, item in items)
            / examples
            for key in ("l1", "cosine_distance")
        }
        for name, items in totals.items()
    }
    metrics["prediction"]["copy_normalized_gain"] = 1.0 - (
        metrics["prediction"]["l1"] / max(metrics["copy_state"]["l1"], 1.0e-12)
    )
    paired_query_effect = {
        key: value / examples for key, value in paired_query_totals.items()
    }
    atomic_write_json(
        args.output,
        {
            "schema_version": "mimic-vla-jepa-evaluation-v1",
            "checkpoint": str(args.checkpoint.resolve()),
            "features": str(args.features.resolve()),
            "checkpoint_step": int(checkpoint["step"]),
            "examples": examples,
            "split": args.split,
            "metrics": metrics,
            "paired_query_effect": paired_query_effect,
        },
    )
    print({"metrics": metrics, "paired_query_effect": paired_query_effect})


if __name__ == "__main__":
    main()
