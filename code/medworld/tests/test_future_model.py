"""Exercise source isolation, matched heads, and all five actual loss paths."""

from unittest.mock import patch

from PIL import Image
import pytest
import torch
from torch import nn

from medworld.config import load_config
from medworld.downstream_tasks.future import FUTURE_TASKS
from medworld.model import MedWorld


def config(**changes):
    cfg = load_config(overrides={"slot_conditioning": False, "decoder_width": 16,
        "decoder_depth": 1, "vision_pixels": 32, "answer_tokens": 16,
        "context_tokens": 32, "generation_tokens": 8, "latent_weight": 0,
        "predictor_width": 16, "predictor_depth": 1,
        "qwen": "/missing/qwen", "jepa": "/missing/jepa"})
    cfg.update(future_enabled=True, future_report_tokens=24, future_source_bytes=20,
               future_weight=1.0)
    cfg.update(changes)
    return cfg


def batch(task):
    row = {"images": [Image.new("RGB", (32, 32), (120, 70, 30))],
           "reports": ["Source chest report. SECRET_OUTSIDE_BUDGET"],
           "delta_hours": torch.tensor([24.0])}
    if task == "future_vqa":
        row.update(questions=["Pleural effusion?"], answers=["yes"])
    elif task == "progression":
        row.update(questions=["Left lung edema change?"], labels=torch.tensor([2]))
    elif task == "future_report":
        row.update(answers=["Future edema."])
    elif task == "mortality_30d":
        row.update(labels=torch.tensor([1.0]), delta_hours=torch.tensor([720.0]))
    else:
        row.update(labels=torch.tensor([3.5]))
    return row


def test_raw_baseline_all_future_losses_predictions_and_checkpoint():
    with patch.object(MedWorld, "_init_slot_branch", side_effect=AssertionError("VLM loaded")):
        model = MedWorld(config(), "cpu")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for task in FUTURE_TASKS:
            optimizer.zero_grad(set_to_none=True)
            loss, parts = model(task, model.prepare_batch(task, batch(task)))
            assert torch.isfinite(loss)
            assert set(parts) == {task}
            loss.backward()
            assert model.task_decoder.patch_projection.weight.grad.abs().sum() > 0
            assert model.future.time[0].weight.grad.abs().sum() > 0
            if task in ("future_vqa", "progression"):
                assert model.future.question_embedding.weight.grad.abs().sum() > 0
            optimizer.step()
        assert not hasattr(model, "encoder") and not hasattr(model, "world")
        saved = model.compact_state()
        restored = MedWorld(config(), "cpu")
        restored.restore(saved)
        model.eval(); restored.eval()
        for task in FUTURE_TASKS:
            actual, expected = model.predict(task, batch(task)), restored.predict(task, batch(task))
            if isinstance(actual, torch.Tensor):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            else:
                assert actual == expected
        assert model.predict("future_vqa", batch("future_vqa"))[0] in model.future.vocabulary
        assert model.predict("remaining_los", batch("remaining_los")).item() >= 0
        assert 0 <= model.predict("mortality_30d", batch("mortality_30d")).item() <= 1


class RecordingWorld(nn.Module):
    def __init__(self):
        super().__init__()
        self.slope = nn.Parameter(torch.linspace(-.001, .001, 1024))

    def forward(self, source, hours):
        return source + hours[:, None, None] * self.slope


def fake_slot_branch(model):
    model.encoder = nn.Module()
    model.encoder.slot_queries = nn.Parameter(torch.randn(8, 1024))
    model.world = RecordingWorld()


def test_future_slots_use_source_and_requested_time_with_identical_task_initialization():
    raw = MedWorld(config(), "cpu")
    with patch.object(MedWorld, "_init_slot_branch", fake_slot_branch):
        slots = MedWorld(config(slot_conditioning=True), "cpu")
    assert raw.metadata["task_initialization_sha256"] == slots.metadata["task_initialization_sha256"]
    for name, value in raw.named_parameters():
        torch.testing.assert_close(value, dict(slots.named_parameters())[name], rtol=0, atol=0)
    row = batch("progression")
    state = slots.encoder.slot_queries[None]
    with patch.object(slots, "encode", return_value=state) as encode:
        features, predicted = slots.future_inputs(row)
        assert encode.call_args.kwargs["texts"] == ["Source chest report."]
        torch.testing.assert_close(predicted, state + row["delta_hours"][:, None, None] * slots.world.slope)
        changed = {**row, "answers": ["LEAKED FUTURE REPORT"], "labels": torch.tensor([0]),
                   "target": {"images": ["not-an-image"], "reports": ["target secret"]}}
        actual, _ = slots.future_inputs(changed)
        torch.testing.assert_close(features, actual, rtol=0, atol=0)
        later, _ = slots.future_inputs({**row, "delta_hours": torch.tensor([96.0])})
        assert not torch.equal(features, later)
        loss, _ = slots("progression", row)
        loss.backward()
        assert slots.encoder.slot_queries.grad.abs().sum() > 0
        assert slots.world.slope.grad.abs().sum() > 0


def test_future_prediction_does_not_require_targets_and_uses_questions():
    model = MedWorld(config(), "cpu").eval()
    row = batch("progression")
    source = {key: value for key, value in row.items() if key != "labels"}
    assert model.predict("progression", source).shape == (1,)
    features, _ = model.future_inputs(source)
    first = model.future.logits("progression", features, row["delta_hours"], ["Left edema?"])
    second = model.future.logits("progression", features, row["delta_hours"], ["Right effusion?"])
    assert not torch.equal(first, second)
    with pytest.raises(ValueError, match="question"):
        model.predict("progression", {key: value for key, value in source.items() if key != "questions"})
    for hours in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="horizons"):
            model.predict("progression", {**source, "delta_hours": torch.tensor([hours])})


def test_future_label_validation_and_disabled_heads():
    model = MedWorld(config(), "cpu")
    with pytest.raises(ValueError, match="official categorical"):
        model("future_vqa", {**batch("future_vqa"), "answers": [["yes", "no"]]})
    with pytest.raises(ValueError, match="integer"):
        model("progression", {**batch("progression"), "labels": torch.tensor([1.0])})
    with pytest.raises(ValueError, match="binary"):
        model("mortality_30d", {**batch("mortality_30d"), "labels": torch.tensor([2.0])})
    with pytest.raises(ValueError, match="nonnegative"):
        model("remaining_los", {**batch("remaining_los"), "labels": torch.tensor([-1.0])})
    for task in ("mortality_30d", "remaining_los"):
        incorrect = {**batch(task), "delta_hours": torch.tensor([72.0])}
        with pytest.raises(ValueError, match="fixed"):
            model(task, incorrect)
        with pytest.raises(ValueError, match="fixed"):
            model.predict(task, incorrect)
    disabled = MedWorld(config(future_enabled=False), "cpu")
    with pytest.raises(ValueError, match="disabled"):
        disabled.predict("future_vqa", batch("future_vqa"))


def test_future_report_output_obeys_shared_utf8_budget_even_without_eos():
    model = MedWorld(config(), "cpu").eval()
    with patch("medworld.downstream_tasks.future.heads.generate_cached", return_value=["�" * 100]):
        prediction = model.predict("future_report", batch("future_report"))[0]
    assert len(prediction.encode("utf-8")) <= model.cfg["future_report_tokens"] - 1
    assert prediction == "�" * 7
