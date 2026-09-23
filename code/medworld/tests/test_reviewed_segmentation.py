"""Reviewed channel masks and MRI volume-level evaluation invariants."""

import numpy as np
import pytest
import torch
from itertools import groupby
from types import SimpleNamespace

from medworld.downstream_tasks.segmentation.loss import segmentation_loss
from medworld.downstream_tasks.segmentation.metrics import (
    aggregate_segmentation,
    segmentation_metrics,
)


def test_inactive_channels_and_padding_receive_zero_segmentation_gradient():
    prediction = torch.zeros(2, 6, 6, 8, requires_grad=True)
    target = torch.ones_like(prediction)
    mask = torch.zeros_like(prediction)
    mask[0, :2, 1:5, 2:6] = 1
    mask[1, 2:, 1:5, 2:6] = 1
    loss = segmentation_loss(prediction, target, mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.count_nonzero(prediction.grad[mask == 0]).item() == 0
    assert torch.all(prediction.grad[mask == 1] < 0)


def test_unannotated_logits_and_targets_cannot_change_loss():
    prediction = torch.zeros(2, 6, 4, 4)
    target = torch.ones_like(prediction)
    mask = torch.zeros_like(prediction)
    mask[0, :2] = 1
    mask[1, 2:] = 1
    expected = segmentation_loss(prediction, target, mask)
    changed_prediction, changed_target = prediction.clone(), target.clone()
    changed_prediction[mask == 0] = 50
    changed_target[mask == 0] = 0
    assert segmentation_loss(changed_prediction, changed_target, mask) == expected
    # A four-region MRI sample has the same sample weight as a two-organ CXR.
    individual = [segmentation_loss(prediction[i:i + 1], target[i:i + 1], mask[i:i + 1])
                  for i in range(2)]
    assert torch.allclose(expected, torch.stack(individual).mean())


def test_unsupervised_segmentation_sample_is_rejected():
    tensor = torch.zeros(2, 6, 3, 3)
    mask = torch.zeros_like(tensor)
    mask[0, 0] = 1
    with pytest.raises(ValueError, match="Every segmentation sample"):
        segmentation_loss(tensor, tensor, mask)


def test_metrics_ignore_unannotated_organs_and_include_only_active_counts():
    logits = torch.full((1, 6, 4, 4), 10.)
    targets = torch.zeros_like(logits)
    targets[:, 2:4] = 1
    mask = torch.zeros_like(logits)
    mask[:, 2:4] = 1
    metrics = segmentation_metrics(logits, targets, mask)
    assert metrics["active_channels"] == [2, 3]
    assert metrics["dice_per_organ"] == [1., 1.]
    assert metrics["iou_per_organ"] == [1., 1.]
    assert metrics["intersection_per_organ"] == [16., 16.]


def test_mri_slices_are_summed_per_volume_before_macro_averaging():
    def row(identity, volume, intersection, prediction, target, dataset="ucsf_alptdg"):
        return {
            "id": identity, "volume_id": volume, "patient": volume,
            "dataset": dataset, "active_channels": [2], "target_names": ["NETC"],
            "intersection_per_organ": [intersection],
            "prediction_pixels_per_organ": [prediction],
            "target_pixels_per_organ": [target],
        }

    records = [
        row("v1:z0", "v1", 100, 100, 100),
        row("v1:z1", "v1", 0, 1, 1),
        row("v2:z0", "v2", 0, 10, 10),
        row("m1:z0", "m1", 3, 3, 3, dataset="mu_glioma_post"),
    ]
    summary = aggregate_segmentation(records)
    ucsf = summary["by_dataset"]["ucsf_alptdg"]
    expected_dice = ((200 + 1e-6) / (202 + 1e-6) + 1e-6 / (20 + 1e-6)) / 2
    expected_iou = ((100 + 1e-6) / (102 + 1e-6) + 1e-6 / (20 + 1e-6)) / 2
    assert ucsf["n_images"] == 3
    assert ucsf["n_volumes"] == 2
    assert ucsf["n_patients"] == 2
    assert ucsf["mean_dice"] == pytest.approx(expected_dice)
    assert ucsf["mean_iou"] == pytest.approx(expected_iou)
    assert summary["mean_dice"] == pytest.approx((expected_dice + 1) / 2)
    assert summary["mean_iou"] == pytest.approx((expected_iou + 1) / 2)
    assert not np.isclose(ucsf["mean_dice"], 1 / 3), "Must not average individual slice Dice"


def test_volume_aggregation_rejects_inconsistent_channel_semantics():
    base = {"patient": "patient", "dataset": "mri", "active_channels": [2],
            "target_names": ["NETC"], "intersection_per_organ": [0],
            "prediction_pixels_per_organ": [0], "target_pixels_per_organ": [0]}
    with pytest.raises(ValueError, match="Channel semantics"):
        aggregate_segmentation([
            dict(base, id="z0", volume_id="volume"),
            dict(base, id="z1", volume_id="volume", active_channels=[3], target_names=["SNFH"]),
        ])


def test_balanced_mri_stream_keeps_volume_blocks_and_exact_resume():
    from medworld.datasets.unified import UnifiedData

    records = [dict(dataset="mimic_cxr_human", kind="human_cxr", volume_id=f"cxr:{i}")
               for i in range(2)]
    for dataset in ("ucsf_alptdg", "mu_glioma_post"):
        for z in range(4):
            for volume in range(3):
                records.append(dict(dataset=dataset, kind="mri", volume_id=f"{dataset}:{volume}", slice_index=z))

    def sampler():
        data = UnifiedData.__new__(UnifiedData)
        data.future = None
        data.current = SimpleNamespace(_records={"segmentation": {"train": records}})
        data._segmentation_sampling = "balanced_dataset"
        data._segmentation_groups = {}
        for index, row in enumerate(records):
            data._segmentation_groups.setdefault(row["dataset"], []).append(index)
        data._permutations = {}
        data.batch = lambda task, split, indices: indices
        return data

    full = sampler().training_batch("segmentation", 0, 36, 42)
    selected = [records[index] for index in full]
    for dataset in ("mimic_cxr_human", "ucsf_alptdg", "mu_glioma_post"):
        rows = [row for row in selected if row["dataset"] == dataset]
        assert len(rows) == 12
        if dataset != "mimic_cxr_human":
            blocks = [list(group) for _, group in groupby(rows, key=lambda row: row["volume_id"])]
            assert len(blocks) == 3
            assert all(len(block) == 4 for block in blocks)
            assert all({row["slice_index"] for row in block} == set(range(4)) for block in blocks)
    assert sampler().training_batch("segmentation", 13, 23, 42) == full[13:]
    assert sampler().training_batch("segmentation", 0, 36, 43) != full
