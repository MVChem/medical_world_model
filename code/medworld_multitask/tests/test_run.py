"""CPU-only checks of checkpoint/resume and patient-safe donor routing."""
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest
import torch
from torch import nn
import torch.nn.functional as F


SPEC = importlib.util.spec_from_file_location("multitask_run", Path(__file__).parents[1] / "run.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class TinyDataset:
    def __init__(self, task, split):
        self.rows = [dict(task=task, split=split, id=str(index), subject_id=str(index // 2),
                          image=index, value=torch.tensor([index / 5, 0.5])) for index in range(7)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class TinyData:
    def __init__(self, **kwargs):
        self.metadata = dict(test_fixture=True)
        self.pos_weight = torch.ones(13)

    def dataset(self, task, split):
        return TinyDataset(task, split)

    @staticmethod
    def collate(task, examples):
        return dict(task=task, split=examples[0]["split"],
                    ids=[item["id"] for item in examples],
                    subject_ids=[item["subject_id"] for item in examples],
                    images=[item["image"] for item in examples],
                    value=torch.stack([item["value"] for item in examples]))


class TinyModel(nn.Module):
    def __init__(self, model_id, condition, device):
        super().__init__()
        self.condition = condition
        self.frozen = nn.Parameter(torch.ones(2), requires_grad=False)
        for task in runner.TASKS:
            setattr(self, task, nn.Linear(2, 1))
        self.metadata = dict(model_id=model_id, condition=condition, frozen_reference="test fixture")

    def set_pos_weight(self, value):
        pass

    def loss(self, task, batch):
        if self.condition == "shuffled":
            assert "donor_images" in batch
        values = F.dropout(batch["value"], p=0.2, training=self.training)
        return (getattr(self, task)(values) - 2).square().mean()

    def predict(self, task, batch, max_new_tokens=16):
        values = getattr(self, task)(batch["value"])
        return [f"test {float(value):.6f}" for value in values[:, 0]] if task == "report" else values


@pytest.fixture
def fake_runtime(monkeypatch):
    monkeypatch.setitem(sys.modules, "data", types.SimpleNamespace(MultiTaskData=TinyData))
    monkeypatch.setitem(sys.modules, "model", types.SimpleNamespace(MultiTaskModel=TinyModel))


def invoke(out, *arguments):
    return runner.main(["--out", str(out), "--device", "cpu", "--validation-batches", "1",
                        "--validate-every", "4", "--save-every", "4", *arguments])


def test_resume_reproduces_uninterrupted_optimizer_and_sampler(tmp_path, fake_runtime):
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    assert invoke(full, "--steps", "8", "--accumulation", "2") == 0
    assert invoke(resumed, "--steps", "4", "--accumulation", "2") == 0
    checkpoint = resumed / "checkpoint_latest.pt"
    assert invoke(resumed, "--steps", "8", "--accumulation", "2", "--resume", str(checkpoint)) == 0
    first = torch.load(full / "checkpoint_latest.pt", weights_only=False)
    second = torch.load(checkpoint, weights_only=False)
    assert first["streams"] == second["streams"]
    assert first["last_validation"] == second["last_validation"]
    for name in first["model_trainable"]:
        assert torch.equal(first["model_trainable"][name], second["model_trainable"][name])
    assert "frozen" not in first["model_trainable"]
    for identity, state in first["optimizer"]["state"].items():
        for key, value in state.items():
            assert torch.equal(value, second["optimizer"]["state"][identity][key])


def test_smoke_reloads_outputs_and_is_not_table_ready(tmp_path, fake_runtime):
    assert invoke(tmp_path / "smoke", "--smoke", "--condition", "shuffled") == 0
    checks = json.loads((tmp_path / "smoke" / "smoke_checks.json").read_text())
    assert checks["checkpoint_reload_exact"]
    assert checks["smoke_only"] and not checks["table2_ready"]
    assert checks["optimizer_steps"] == 4
    assert checks["predictions"]["report"]["reload_output_equal"]
    assert set(checks["decoder_initial_sha256"]) == {"segmentation", "sr"}
    with pytest.raises(ValueError, match="configuration differs"):
        invoke(tmp_path / "training", "--steps", "8", "--condition", "shuffled",
               "--resume", str(tmp_path / "smoke" / "checkpoint_latest.pt"))


def test_image_only_is_explicitly_dense_only(tmp_path, fake_runtime):
    assert invoke(tmp_path / "smoke", "--smoke", "--condition", "image_only") == 0
    checks = json.loads((tmp_path / "smoke" / "smoke_checks.json").read_text())
    assert checks["tasks"] == ["segmentation", "sr"]
    assert checks["optimizer_steps"] == 2


def test_donor_is_different_patient_same_split_and_stable_at_batch_one():
    data = TinyData()
    dataset = data.dataset("sr", "validate")
    batch = data.collate("sr", [dataset[0]])
    first = runner.add_donors(batch, dataset, seed=42)["donor_images"]
    again = runner.add_donors(data.collate("sr", [dataset[0]]), dataset, seed=42)["donor_images"]
    assert first == again
    assert dataset[first[0]]["subject_id"] != dataset[0]["subject_id"]
    assert dataset[first[0]]["split"] == "validate"
    for row in dataset.rows:
        row["subject_id"] = "same"
    with pytest.raises(ValueError, match="two different patients"):
        runner.add_donors(data.collate("sr", [dataset[0]]), dataset, seed=42)


def test_populated_output_requires_resume(tmp_path, fake_runtime):
    (tmp_path / "keep.txt").write_text("existing results")
    with pytest.raises(FileExistsError):
        invoke(tmp_path, "--smoke")
    assert (tmp_path / "keep.txt").read_text() == "existing results"
