"""Causal input isolation and matched-data controls, using synthetic images."""
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch


MODULE = Path(__file__).resolve().parents[1] / "data.py"
spec = importlib.util.spec_from_file_location("native_forecast_data_test_module", MODULE)
data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data)


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self):
        self.seen = []

    def encode(self, text, add_special_tokens=False):
        self.seen.append(text)
        return [ord(char) + 2 for char in text]


def make_cache(tmp_path, condition="slots"):
    observations, pairs = [], {}
    for split in data.SPLITS:
        pairs[split] = []
        for patient in range(2):
            ids = []
            for time in range(2):
                oid = f"{split}-{patient}-{time}"
                ids.append(oid)
                image = tmp_path / f"{oid}.png"
                Image.new("RGB", (17, 31), (30 + 60 * patient, 60 + 80 * time, 100)).save(image)
                observations.append(dict(id=oid, patient=f"{split}-{patient}", split=split,
                                         image=str(image), report=f"report {oid}",
                                         ehr_text=f"ehr {oid}", labels=[patient, time]))
            pairs[split].append(dict(id=f"pair-{split}-{patient}", patient=f"{split}-{patient}",
                                     split=split, source=ids[0], target=ids[1], horizon=patient))
    for name, rows in {"observations": observations, **pairs}.items():
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    np.save(tmp_path / "vjepa_features.npy", np.arange(len(observations) * 6, dtype=np.float16).reshape(-1, 2, 3))
    cfg = dict(cache=str(tmp_path), report_tokens=12, ehr_tokens=100, use_ehr=True,
               batch_size=3, gradient_accumulation=2, seed=42, sampling="permutation", state_condition=condition)
    return cfg


class FutureForbidden(dict):
    def __getitem__(self, key):
        if key in {"image", "report", "ehr_text", "labels"}:
            raise AssertionError(f"Read future {key} during prediction")
        return super().__getitem__(key)


class GuardFeatures:
    def __init__(self, features, forbidden):
        self.features = features
        self.forbidden = forbidden

    def __getitem__(self, indices):
        assert not self.forbidden.intersection(indices), "Read target feature during prediction"
        return self.features[indices]


def test_source_only_never_reads_future_and_whitelist_removes_supervision(tmp_path):
    corpus = data.Corpus(make_cache(tmp_path), Tokenizer())
    rows = corpus.pairs["test"]
    target_indices = {corpus.lookup[row["target"]] for row in rows}
    for index in target_indices:
        corpus.observations[index] = FutureForbidden(corpus.observations[index])
    corpus.features = GuardFeatures(corpus.features, target_indices)
    # Even the pair's target pointer is unnecessary at inference.
    inputs = corpus.batch([{k: v for k, v in row.items() if k != "target"} for row in rows],
                          device="cpu", source_only=True)
    assert set(inputs) == {"source_ids", "source_mask", "source_features", "_source_images", "horizon"}
    assert all(image.size == (512, 512) for image in inputs["_source_images"])
    with_targets = {**inputs, "target_features": object(), "target_ids": object(),
                    "_target_images": object(), "source_labels": object(), "source_target_ids": object(),
                    "unexpected_future_ehr": object()}
    assert data.source_view(with_targets).keys() == inputs.keys()


def test_shuffled_donor_preserves_native_decoder_evidence(tmp_path):
    cfg = make_cache(tmp_path, "shuffled")
    shuffled = data.Corpus(cfg, Tokenizer())
    native = data.Corpus({**cfg, "state_condition": "slots"}, Tokenizer())
    for split in data.SPLITS:
        rows = shuffled.pairs[split]
        batch = shuffled.batch(rows, device="cpu", source_only=True)
        original = native.batch(rows, device="cpu", source_only=True)
        for key in ("source_ids", "source_mask", "source_features", "horizon"):
            torch.testing.assert_close(batch[key], original[key])
        for index, row in enumerate(rows):
            donor = shuffled.observations[shuffled.lookup[shuffled.donors[row["id"]]]]
            assert donor["patient"] != row["patient"] and donor["split"] == split
            assert donor["id"] in {pair["source"] for pair in rows}
            assert batch["_source_images"][index].tobytes() == original["_source_images"][index].tobytes()
            assert batch["donor_images"][index].tobytes() != original["_source_images"][index].tobytes()


def test_report_target_excludes_ehr_and_budgets_are_independent(tmp_path):
    tokenizer = Tokenizer()
    corpus = data.Corpus(make_cache(tmp_path), tokenizer)
    rows = corpus.pairs["train"][:1]
    batch = corpus.batch(rows, device="cpu")
    obs = corpus.observations[corpus.lookup[rows[0]["source"]]]
    expected_report = tokenizer.encode(obs["report"])[:12]
    assert batch["source_target_ids"][0].tolist() == expected_report + [1]
    assert batch["source_ids"].shape[1] > 12
    context = batch["source_ids"][0].tolist()
    ehr = tokenizer.encode(obs["ehr_text"])
    assert context[-len(ehr):] == ehr
    corpus = data.Corpus({**corpus.cfg, "ehr_tokens": 1}, Tokenizer())
    with pytest.raises(ValueError, match="independent whole-line"):
        corpus.batch(rows, device="cpu", source_only=True)


def test_resumed_permutation_matches_original_stream_across_epochs(tmp_path):
    cfg = make_cache(tmp_path)
    continuous, resumed = data.Corpus(cfg, Tokenizer()), data.Corpus(cfg, Tokenizer())
    # Observe selections directly; this tests both stage populations and epoch boundaries.
    for corpus in (continuous, resumed):
        corpus.batch = lambda rows, device="cuda", stage1=False, source_only=False: (rows, stage1, device)
    for stage in (1, 2):
        for step in range(4):
            for micro in range(2):
                got = continuous.training_batch(stage, step, micro, device="cpu")
                if step >= 2:
                    assert got == resumed.training_batch(stage, step, micro, device="cpu")
                population = continuous.stage1 if stage == 1 else continuous.pairs["train"]
                start = (step * cfg["gradient_accumulation"] + micro) * cfg["batch_size"]
                indices = []
                for pos in range(start, start + cfg["batch_size"]):
                    epoch, offset = divmod(pos, len(population))
                    indices.append(np.random.default_rng(np.random.SeedSequence([42, stage, epoch])).permutation(len(population))[offset])
                assert got == ([population[index] for index in indices], stage == 1, "cpu")


def test_cross_split_patients_and_stale_features_are_rejected(tmp_path):
    cfg = make_cache(tmp_path)
    path = tmp_path / "observations.jsonl"
    original = path.read_text()
    rows = [json.loads(line) for line in original.splitlines()]
    rows[-1]["patient"] = rows[0]["patient"]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="Patient leakage"):
        data.Corpus(cfg, Tokenizer())
    path.write_text(original)
    (tmp_path / "features.json").write_text(json.dumps({"observations_sha256": "incorrect"}))
    with pytest.raises(ValueError, match="manifest SHA256 mismatch"):
        data.Corpus(cfg, Tokenizer())


def test_metadata_counts_and_pad_zero_are_preserved(tmp_path):
    corpus = data.Corpus(make_cache(tmp_path), Tokenizer())
    assert corpus.metadata["counts"]["test"] == {"pairs": 2, "patients": 2}
    assert not any(corpus.metadata["patient_audit"]["cross_split_patient_intersections"].values())
    assert corpus.metadata["frozen_features"]["array_content_sha256_recomputed"] is False
    ids, mask = corpus.pad([[9], [9, 8]], left=True)
    assert ids.tolist() == [[0, 9], [9, 8]]
    assert mask.tolist() == [[0, 1], [1, 1]]
