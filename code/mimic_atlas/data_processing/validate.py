"""Validate published data and a small CPU head, without loading pretrained models."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path

from .build import CODE_ROOT


def validate(config):
    import torch
    from medworld.config import load_config
    from medworld.datasets.unified import UnifiedData

    cfg = load_config(config, overrides={"decoded_image_cache": 0, "image_workers": 1})
    if not cfg["prepared_data"]:
        raise ValueError("Validation requires an explicit prepared_data configuration")
    torch.set_num_threads(1)
    data = UnifiedData(cfg)
    checked_pairs = 0
    for split, pairs in data.temporal.pairs.items():
        for pair in pairs:
            source = data.temporal.lookup[pair["source"]]
            target = data.temporal.lookup[pair["target"]]
            elapsed = (datetime.fromisoformat(target["timestamp"]) -
                       datetime.fromisoformat(source["timestamp"])).total_seconds() / 3600
            if (source["view"] != target["view"] or source["view"] != pair["matched_view"]
                    or not math.isclose(elapsed, pair["realized_gap_hours"], abs_tol=1e-8)
                    or pair["target_order"] <= pair["source_order"]):
                raise ValueError("Temporal view, chronology or realized interval mismatch")
            if pair["pairing"] == "adjacent" and pair["target_order"] != pair["source_order"] + 1:
                raise ValueError("Incorrect adjacency provenance")
            checked_pairs += 1
    decoded = {}
    for task, counts in data.current.counts.items():
        for split, count in counts.items():
            if not count:
                continue
            batch = data.batch(task, split, [0])
            if task == "classification" and batch["labels"].shape != (1, 13):
                raise ValueError("Unexpected classification batch shape")
            if task == "segmentation":
                channels = 6 if data.current.manual_only else 2 if split == "human_test" else 3
                if batch["targets"].shape != (1, channels, 256, 256):
                    raise ValueError("Unexpected segmentation batch shape")
            decoded[f"{task}/{split}"] = len(batch["images"])
    segmentation_groups = {}
    if data.current.manual_only:
        for split, rows in data.current._records["segmentation"].items():
            groups = {}
            for index, row in enumerate(rows):
                if row["kind"] == "cxas" or "target_file" in row:
                    raise ValueError("Pseudo segmentation entered the reviewed dataset")
                groups.setdefault(row["dataset"], []).append(index)
            for dataset, indices in groups.items():
                volumes = {rows[index]["volume_id"] for index in indices}
                patients = {rows[index]["subject_id"] for index in indices}
                segmentation_groups[f"{dataset}/{split}"] = {
                    "images_or_slices": len(indices), "volumes": len(volumes), "patients": len(patients)}
                selected = sorted({indices[0], indices[len(indices) // 2], indices[-1]})
                # Decode separately to avoid retaining many six-channel tensors.
                for index in selected:
                    example = data.batch("segmentation", split, [index])
                    expected = rows[index]["channels"]
                    active = (example["mask"][0].sum((1, 2)) > 0).nonzero().flatten().tolist()
                    if active != expected:
                        raise ValueError("Segmentation active channels changed during decoding")
                decoded[f"{dataset}/{split}"] = len(selected)
        required = {f"{dataset}/{split}" for dataset in
                    ("mimic_cxr_human", "ucsf_alptdg", "mu_glioma_post")
                    for split in ("train", "validate", "test")}
        required.add("montgomery/human_test")
        if not required.issubset(segmentation_groups):
            raise ValueError(f"Required reviewed segmentation datasets/splits missing: {sorted(required - segmentation_groups.keys())}")
        # Exercise mixed-modality collation and real six-channel head gradients,
        # without loading a foundation model or performing an optimizer update.
        from medworld.downstream_tasks.segmentation.decoder import SegmentationHead
        from medworld.downstream_tasks.segmentation.loss import segmentation_loss
        from medworld.downstream_tasks.segmentation.metrics import segmentation_metrics
        batch = data.training_batch("segmentation", 0, 3, cfg["seed"])
        head = SegmentationHead(width=32, channels=6)
        prediction = head(torch.randn(3, 16, 32))
        loss = segmentation_loss(prediction, batch["targets"], batch["mask"])
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite mixed segmentation loss")
        loss.backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()):
            raise ValueError("Nonfinite or missing six-channel head gradients")
        for i in range(3):
            segmentation_metrics(prediction[i:i+1].detach(), batch["targets"][i:i+1], batch["mask"][i:i+1])
        decoded["mixed_segmentation_head_forward_backward"] = 3
    for split, pairs in data.temporal.pairs.items():
        if not pairs:
            continue
        data.batch("temporal", split, [0])
        inference = data.batch("temporal", split, [0], source_only=True)
        if set(inference) != {"source", "delta_hours"}:
            raise ValueError("Source-only temporal boundary includes unexpected fields")
        decoded[f"temporal/{split}"] = 1
    for split in ("train", "validate", "test", "human_test"):
        if not data.current.counts["segmentation"][split]:
            raise ValueError(f"Required segmentation split empty: {split}")
    return {
        "dataset": cfg["prepared_data"], "fingerprint": data.fingerprint,
        "counts_after_global_holdouts": data.current.counts,
        "temporal_pairs": {s: len(rows) for s, rows in data.temporal.pairs.items()},
        "temporal_directed_pairs": {s: len(data.rows("temporal", s)) for s in data.temporal.pairs},
        "checked_temporal_pairs": checked_pairs, "decoded_sample_batches": decoded,
        "segmentation_by_dataset": segmentation_groups,
        "globally_patient_disjoint": data.metadata["globally_patient_disjoint"],
        "dropped_current_rows": data.metadata["current_dropped_rows"],
        "no_foundation_model_or_training_started": True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CODE_ROOT / "medworld/configs/medworld_0922.json")
    parser.add_argument("--run-dir", type=Path, default=Path(__file__).resolve().parent / "runs" /
                        ("validate_" + datetime.now().strftime("%Y%m%d")))
    args = parser.parse_args(argv)
    result = validate(args.config)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
