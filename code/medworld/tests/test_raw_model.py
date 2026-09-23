"""Verify branch isolation and common initialization through the real task model."""
from unittest.mock import patch

from PIL import Image
import pytest
import torch
from torch import nn

from medworld.config import load_config
from medworld.model import MedWorld


def config(**changes):
    return load_config(overrides={
        "architecture": "raw_input_v1", "slot_conditioning": False,
        "decoder_width": 16, "decoder_depth": 1, "vision_pixels": 32,
        "answer_tokens": 16, "context_tokens": 32, "generation_tokens": 8,
        "latent_weight": 0, "visual_consistency_weight": 0,
        "qwen": "/does/not/exist/qwen", "jepa": "/does/not/exist/jepa",
        **changes,
    })


def classification_batch():
    return {"images": [Image.new("RGB", (32, 32), (130, 40, 80))],
            "labels": torch.ones(1, 13), "label_mask": torch.ones(1, 13, dtype=torch.bool)}


def test_baseline_never_loads_pretrained_assets_or_constructs_slot_branch():
    with patch.dict("sys.modules", {"transformers": None}), \
         patch.object(MedWorld, "_init_slot_branch", side_effect=AssertionError("slot branch loaded")):
        model = MedWorld(config(visual_consistency_weight=.1, latent_weight=1), "cpu")
        assert model.cfg["latent_weight"] == model.cfg["visual_consistency_weight"] == 0
        assert model.target is None
        assert all(not hasattr(model, key) for key in (
            "encoder", "world", "jepa", "visual_teacher", "visual_reconstruction", "processor", "tokenizer"))
        batch = classification_batch()
        loss, parts = model("classification", model.prepare_batch("classification", batch))
        loss.backward()
        assert set(parts) == {"classification"}
        assert model.task_decoder.patch_projection.weight.grad.abs().sum() > 0
        assert not any("lora" in name for name, _ in model.named_parameters())
        with pytest.raises(ValueError, match="no slot encoder"):
            model.encode(batch["images"])
        with pytest.raises(ValueError, match="no temporal"):
            model("temporal", {})


def fake_slot_branch(model):
    model.encoder = nn.Module()
    model.encoder.slot_queries = nn.Parameter(torch.randn(8, 1024))


def test_slots_add_conditions_without_changing_raw_task_initialization():
    baseline = MedWorld(config(), "cpu")
    with patch.object(MedWorld, "_init_slot_branch", fake_slot_branch):
        slots = MedWorld(config(slot_conditioning=True), "cpu")
    assert baseline.metadata["task_initialization_sha256"] == slots.metadata["task_initialization_sha256"]
    other = dict(slots.named_parameters())
    for name, value in baseline.named_parameters():
        torch.testing.assert_close(value, other[name], rtol=0, atol=0)
    batch = classification_batch()
    state = slots.encoder.slot_queries[None]
    with patch.object(slots, "encode", return_value=state) as encode:
        conditioned, actual = slots.task_inputs(batch["images"])
        encode.assert_called_once()
        assert actual is state
        direct, _ = baseline.task_inputs(batch["images"])
        assert not torch.equal(conditioned, direct)
        conditioned.square().mean().backward()
        assert slots.encoder.slot_queries.grad.abs().sum() > 0
        assert slots.task_decoder.patch_projection.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="report-only slots are not defined"):
        slots.task_inputs(reports=["normal lungs"])


def test_raw_baseline_three_tasks_and_checkpoint_roundtrip():
    model = MedWorld(config(), "cpu")
    image = classification_batch()["images"]
    batches = {
        "classification": classification_batch(),
        "segmentation": {"images": image, "targets": torch.ones(1, 6, 256, 256),
                         "mask": torch.ones(1, 6, 256, 256)},
        "vqa": {"images": image, "questions": ["Normal?"], "answers": [["yes"]]},
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for task, batch in batches.items():
        optimizer.zero_grad(set_to_none=True)
        loss, parts = model(task, batch)
        assert torch.isfinite(loss)
        assert set(parts) == {task}
        loss.backward()
        optimizer.step()
    saved = model.compact_state()
    assert saved["ema"] is None
    restored = MedWorld(config(), "cpu")
    restored.restore(saved)
    model.eval(); restored.eval()
    for task in ("classification", "segmentation"):
        torch.testing.assert_close(model.predict(task, batches[task]), restored.predict(task, batches[task]), rtol=0, atol=0)
    assert model.predict("vqa", batches["vqa"]) == restored.predict("vqa", batches["vqa"])
    saved["ema"] = {}
    with pytest.raises(ValueError, match="must not contain EMA"):
        restored.restore(saved)


def test_report_input_is_explicit_and_segmentation_requires_image():
    model = MedWorld(config(), "cpu").eval()
    plain = classification_batch()
    reported = {**plain, "reports": ["No pleural effusion."]}
    assert not torch.equal(model.predict("classification", plain), model.predict("classification", reported))
    assert model.predict("classification", {"reports": ["Normal chest."]}).shape == (1, 13)
    with pytest.raises(ValueError, match="Segmentation requires"):
        model.predict("segmentation", {"reports": ["Normal chest."]})
    with pytest.raises(ValueError, match="Segmentation requires"):
        model("segmentation", {"reports": ["Normal chest."]})
