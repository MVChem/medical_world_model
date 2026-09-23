"""Per-image, per-organ binary Dice and intersection-over-union metrics."""
import numpy as np


def segmentation_metrics(prediction, target, mask):
    if len(target) != 1:
        raise ValueError("Segmentation evaluation expects one image at a time")
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError("Segmentation predictions, targets and channel masks must align")
    active = mask[0].sum((1, 2)) > 0
    if not active.any():
        raise ValueError("No supervised segmentation channels")
    p = (prediction.sigmoid() > .5).float() * mask
    t = (target > .5).float() * mask
    dice = ((2 * (p * t).sum((2, 3)) + 1e-6) /
            (p.sum((2, 3)) + t.sum((2, 3)) + 1e-6))[0][active].tolist()
    intersection = (p * t).sum((2, 3))
    union = p.sum((2, 3)) + t.sum((2, 3)) - intersection
    # Match Dice's existing convention: two empty masks score 1.
    iou = ((intersection + 1e-6) / (union + 1e-6))[0][active].tolist()
    return {"dice_per_organ": dice, "mean_dice": float(np.mean(dice)),
            "iou_per_organ": iou, "mean_iou": float(np.mean(iou)),
            "active_channels": active.nonzero().flatten().tolist(),
            "intersection_per_organ": intersection[0][active].tolist(),
            "prediction_pixels_per_organ": p.sum((2, 3))[0][active].tolist(),
            "target_pixels_per_organ": t.sum((2, 3))[0][active].tolist()}


def aggregate_segmentation(records):
    """Sum MRI slice counts within each volume, then report each source separately."""
    from collections import defaultdict
    if not records:
        raise ValueError("Cannot aggregate empty segmentation results")
    groups = defaultdict(list)
    for row in records:
        if any(key not in row for key in ("dataset", "volume_id", "target_names")):
            raise ValueError("Segmentation records require dataset, volume_id and target_names")
        groups[row["dataset"]].append(row)
    summaries = {}
    for dataset, rows in groups.items():
        volumes = {}
        channels = rows[0]["active_channels"]
        names = rows[0]["target_names"]
        for row in rows:
            if row["active_channels"] != channels or row["target_names"] != names:
                raise ValueError("Channel semantics changed within a segmentation dataset")
            identity = row["volume_id"]
            totals = volumes.setdefault(identity, np.zeros((3, len(channels)), dtype=np.float64))
            totals += np.asarray([row["intersection_per_organ"], row["prediction_pixels_per_organ"],
                                  row["target_pixels_per_organ"]], dtype=np.float64)
        values = np.stack(list(volumes.values()))
        intersection, prediction, target = values[:, 0], values[:, 1], values[:, 2]
        dice = (2 * intersection + 1e-6) / (prediction + target + 1e-6)
        iou = (intersection + 1e-6) / (prediction + target - intersection + 1e-6)
        summaries[dataset] = {
            "n_images": len(rows), "n_volumes": len(volumes),
            "n_patients": len({r["patient"] for r in rows}),
            "active_channels": channels, "target_names": names,
            "mean_dice": float(dice.mean()), "mean_iou": float(iou.mean()),
            "dice_per_organ": dice.mean(0).tolist(), "iou_per_organ": iou.mean(0).tolist(),
        }
    result = {
        "mean_dice": float(np.mean([v["mean_dice"] for v in summaries.values()])),
        "mean_iou": float(np.mean([v["mean_iou"] for v in summaries.values()])),
        "by_dataset": summaries, "threshold": 0.5, "empty_union_score": 1.0,
        "aggregation": "Sum selected axial slice intersections within MRI volume; mean over annotated organs/volumes per dataset; macro mean over datasets",
    }
    if len(summaries) == 1:
        entry = next(iter(summaries.values()))
        result.update({k: entry[k] for k in ("dice_per_organ", "iou_per_organ", "target_names")})
    return result
