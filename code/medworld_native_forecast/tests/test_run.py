"""Execution contracts exercised with a stochastic CPU model, without base weights."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import types

import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("native_forecast_run_tested", ROOT / "run.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class FakeCorpus:
    metadata = {"cohort": "fixed"}

    def __init__(self, cfg, tokenizer):
        self.cfg = dict(cfg)
        self.pairs = {split: [dict(id=f"{split}{i}", source=i, target=i + 100) for i in range(7)]
                      for split in ("train", "validate", "test")}

    def batch(self, rows, device="cpu", source_only=False):
        value = torch.tensor([[float(row["source"] + 1)] for row in rows])
        result = dict(source_ids=value)
        if not source_only:
            result.update(target_ids=value + 7, source_labels=value + 2)
        return result

    def training_batch(self, stage, step, micro, device="cpu"):
        rng = np.random.default_rng(np.random.SeedSequence([self.cfg["seed"], stage, step, micro]))
        return self.batch([self.pairs["train"][int(rng.integers(7))]], device)


class FakeModel(nn.Module):
    metadata = {"model": "stochastic_cpu"}
    tokenizer = None

    def __init__(self, cfg, condition="slots", device="cpu"):
        super().__init__()
        self.condition = condition
        # Decoder initialized first so all branches start with the same weights.
        self.decoder = nn.Linear(1, 1)
        self.finding = nn.Linear(1, 1)
        self.encoder = nn.Linear(1, 1) if condition != "native" else None
        self.target_encoder = None
        self.dropout = nn.Dropout(0.2)

    def begin_stage2(self):
        if self.encoder is not None:
            self.target_encoder = copy.deepcopy(self.encoder).requires_grad_(False)

    def losses(self, batch, stage):
        values = self.dropout(batch["source_ids"])
        output = self.decoder(values) + self.finding(values)
        if self.encoder is not None:
            output = output + self.encoder(values)
        loss = (output - batch["target_ids"]).square().mean()
        return loss, {"text": loss}

    def gradient_audit(self, stage):
        return {name: float(parameter.grad.norm()) for name, parameter in self.named_parameters()
                if parameter.grad is not None}

    def compact_state(self):
        return {name: value.detach().clone() for name, value in self.state_dict().items()}

    def load_compact(self, state):
        if self.encoder is None:
            state = {name: value for name, value in state.items() if name.startswith(("decoder.", "finding."))}
        elif any(name.startswith("target_encoder.") for name in state) and self.target_encoder is None:
            self.begin_stage2()
        self.load_state_dict(state, strict=False)

    def predict(self, batch):
        values = self.decoder(batch["source_ids"])
        return [str(round(float(value), 5)) for value in values[:, 0]], self.finding(values).sigmoid()


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setitem(sys.modules, "model", types.SimpleNamespace(NativeForecast=FakeModel))
    monkeypatch.setitem(sys.modules, "data", types.SimpleNamespace(Corpus=FakeCorpus,
        source_view=lambda batch: {key: value for key, value in batch.items() if key == "source_ids"}))
    path = tmp_path / "config.json"
    path.write_text(json.dumps(dict(seed=42, batch_size=1, gradient_accumulation=2,
        stage1_gradient_accumulation=2, learning_rate=0.01, lora_learning_rate=0.005,
        sampling="permutation", warmup_steps=2, validation_every_steps=0,
        max_stage1_steps=8, max_stage2_steps=8, generation_tokens=8)))
    return path


def invoke(config, out, *arguments):
    return runner.main(["--config", str(config), "--out", str(out), "--device", "cpu", "--threads", "1",
                        "--save-every", "2", *arguments])


def checkpoint(path):
    return torch.load(path / "checkpoint_final.pt", map_location="cpu", weights_only=False)


def assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            assert_tree_equal(left, right)
    else:
        assert a == b


def test_resume_matches_continuous_stochastic_training(setup, tmp_path):
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    invoke(setup, full, "--stage", "stage1", "--steps", "8")
    invoke(setup, resumed, "--stage", "stage1", "--steps", "4")
    invoke(setup, resumed, "--stage", "stage1", "--steps", "8",
           "--resume", str(resumed / "checkpoint_latest.pt"))
    left, right = checkpoint(full), checkpoint(resumed)
    assert_tree_equal(left["model"], right["model"])
    assert_tree_equal(left["optimizer"], right["optimizer"])
    assert_tree_equal(left["sampler"], right["sampler"])
    assert torch.equal(left["rng"]["torch"], right["rng"]["torch"])


def test_smoke_target_freeze_reload_and_future_boundary(setup, tmp_path):
    out = tmp_path / "smoke"
    assert invoke(setup, out, "--smoke") == 0
    checks = json.loads((out / "smoke_checks.json").read_text())
    assert checks["checkpoint_reload_exact"] and checks["future_perturbation_equal"]
    assert checks["target_unchanged"] and checks["target_frozen_hash"]
    assert checkpoint(out)["stage_step"] == 1
    assert checkpoint(out)["sampler"]["accumulation"] == 1


def test_branches_transfer_identical_decoder_and_reject_smoke_to_formal(setup, tmp_path):
    shared = tmp_path / "shared"
    invoke(setup, shared, "--smoke", "--stage", "stage1")
    source = shared / "checkpoint_stage1.pt"
    hashes = []
    for condition in ("native", "slots", "shuffled"):
        out = tmp_path / condition
        invoke(setup, out, "--condition", condition, "--smoke", "--init-checkpoint", str(source))
        hashes.append(json.loads((out / "initialization.json").read_text())["decoder_sha256"])
    assert len(set(hashes)) == 1
    with pytest.raises(ValueError, match="Smoke and formal"):
        invoke(setup, tmp_path / "bad", "--condition", "native", "--init-checkpoint", str(source))


def test_output_guard_and_stage2_requires_initialization(setup, tmp_path):
    out = tmp_path / "already"
    out.mkdir()
    marker = out / "keep.txt"
    marker.write_text("existing user result")
    with pytest.raises(FileExistsError):
        invoke(setup, out, "--smoke")
    assert marker.read_text() == "existing user result"
    with pytest.raises(ValueError, match="requires an explicit shared"):
        invoke(setup, tmp_path / "uninitialized", "--condition", "native", "--smoke")


def test_stage_accumulation_budgets_are_distinct(setup, tmp_path):
    cfg = json.loads(setup.read_text())
    cfg.update(stage1_gradient_accumulation=2, gradient_accumulation=3,
               max_stage1_steps=1, max_stage2_steps=1)
    setup.write_text(json.dumps(cfg))
    out = tmp_path / "both"
    invoke(setup, out)
    events = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert [row["effective_batch_size"] for row in events] == [2, 3]
    saved_stage1 = torch.load(out / "checkpoint_stage1.pt", weights_only=False)
    assert saved_stage1["sampler"]["accumulation"] == 2
    assert checkpoint(out)["sampler"]["accumulation"] == 3


@pytest.mark.parametrize("name", ("run.py", "launch_queue.py", "evaluate.py", "score_results.py", "green_bridge.py"))
def test_cli_fresh_process_does_not_shadow_stdlib(name):
    completed = subprocess.run([sys.executable, str(ROOT / name), "--help"],
        cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout


def test_resume_rejects_source_change(setup, tmp_path):
    out = tmp_path / "pinned"
    invoke(setup, out, "--stage", "stage1", "--steps", "1")
    path = out / "checkpoint_latest.pt"
    saved = torch.load(path, weights_only=False)
    saved["runtime_signature"]["source_sha256"]["model.py"] = "different-source"
    torch.save(saved, path)
    with pytest.raises(ValueError, match="source code or pretrained file identity changed"):
        invoke(setup, out, "--stage", "stage1", "--steps", "2", "--resume", str(path))


@pytest.mark.parametrize("batch", (1, 2))
def test_microbatch_keeps_stage1_eight_stage2_thirtytwo(setup, tmp_path, batch):
    cfg = json.loads(setup.read_text())
    cfg.pop("stage1_gradient_accumulation")
    cfg.update(batch_size=batch, gradient_accumulation=32 // batch)
    setup.write_text(json.dumps(cfg))
    args = runner.parser().parse_args(["--config", str(setup), "--out", str(tmp_path / "unused")])
    effective = runner.effective_config(args)
    assert runner.accumulation_for(effective, 1) * batch == 8
    assert runner.accumulation_for(effective, 2) * batch == 32
    cfg["gradient_accumulation"] = 1.5
    setup.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="whole number"):
        runner.effective_config(args)
