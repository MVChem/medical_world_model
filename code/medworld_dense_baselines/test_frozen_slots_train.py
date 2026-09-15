"""Scientific input-path, frozen-state, and resumability checks for slot probes."""
import argparse
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from common import atomic, digest, write_rows
import frozen_slots_train as training

torch.set_num_threads(2)


def make_fixture(root):
    data_run, run = root / "prepared", root / "run"
    data, cache = data_run / "data", run / "fixture"
    data.mkdir(parents=True)
    cache.mkdir(parents=True)
    rows = []
    for split in ("train", "validate", "test", "human_test"):
        for sample in range(2):
            i = len(rows)
            row = dict(id=f"image{i}", subject_id=f"patient{i}", index=i, split=split,
                       kind="montgomery" if split == "human_test" else "mimic",
                       box=[0, 0, 512, 512], tasks=["segmentation"] if split == "human_test" else ["segmentation", "sr"],
                       old_index=i)
            if split == "human_test":
                row["human_index"] = sample
            rows.append(row)
    write_rows(data / "observations.jsonl", rows)
    rng = np.random.default_rng(1)
    images = rng.integers(0, 256, (len(rows), 512, 512), dtype=np.uint8)
    np.save(data / "images.npy", images)
    np.save(data / "lr_images.npy", images[:, ::4, ::4])
    np.save(data / "human_masks.npy", (images[-2:, None, ::2, ::2] > 100).repeat(2, axis=1).astype(np.uint8))
    pseudo_path = root / "pseudo.npy"
    np.save(pseudo_path, (images[:, None, ::2, ::2] > 100).repeat(3, axis=1).astype(np.float32))
    manifest = dict(cohort_sha256=digest(data / "observations.jsonl"),
                    image_sha256=digest(data / "images.npy"), lr_image_sha256=digest(data / "lr_images.npy"))
    atomic(data / "manifest.json", manifest)
    atomic(cache / "slot_contract.json", dict(**manifest, model_id="fixture", shape=[len(rows), 4, 1024],
          slot_ids=[5, 6, 7, 8], branch_columns=["hr", "lr"], slot_training=False, language_model_loaded=False))
    slots = rng.normal(size=(len(rows), 4, 1024)).astype(np.float16)
    np.save(cache / "hr_slots.npy", slots)
    np.save(cache / "lr_slots.npy", slots + 2)
    np.save(cache / "slots_done.npy", np.ones((len(rows), 2), dtype=bool))
    return data_run, run, pseudo_path, rows


class FrozenSlotTests(unittest.TestCase):
    def test_frozen_slots_and_both_input_paths(self):
        for task in ("segmentation", "sr"):
            torch.manual_seed(7)
            model = training.FrozenSlotHead(task)
            x = torch.rand(1, 1, 128 if task == "sr" else 256, 128 if task == "sr" else 256,
                           requires_grad=True)
            slots = torch.randn(1, 4, 1024, requires_grad=True)
            prediction = model(x, slots)
            self.assertEqual(tuple(prediction.shape), (1, 1, 512, 512) if task == "sr" else (1, 3, 256, 256))
            prediction.square().mean().backward()
            self.assertIsNone(slots.grad, "frozen descriptors must be detached at decoder boundary")
            self.assertGreater(float(x.grad.abs().sum()), 0)
            self.assertGreater(float(model.slot_project[0].weight.grad.abs().sum()), 0)
            self.assertGreater(float(model.attention.in_proj_weight.grad.abs().sum()), 0)
            with torch.no_grad():
                zero_conditioned = model(x, torch.zeros_like(slots))
            self.assertGreater(float((prediction - zero_conditioned).detach().abs().mean()), 1e-6)
            self.assertFalse(any("position" in name for name, _ in model.named_parameters()))

    def test_initialization_is_condition_and_model_independent(self):
        hashes = []
        for _ in ("image_only", "slots", "shuffled_slots"):
            torch.manual_seed(training.DEFAULT_SEED)
            hashes.append(training.state_hash(training.FrozenSlotHead("segmentation")))
        self.assertEqual(len(set(hashes)), 1)

    def test_different_patient_shuffle_without_split_leakage(self):
        rows = [dict(subject_id=f"p{i//2}") for i in range(12)]
        donors = training.patient_derangement(list(range(12)), rows, 7)
        self.assertEqual(set(donors), set(donors.values()))
        self.assertTrue(all(rows[i]["subject_id"] != rows[j]["subject_id"] for i, j in donors.items()))
        self.assertEqual(donors, training.patient_derangement(list(range(12)), rows, 7))
        with self.assertRaises(ValueError):
            training.patient_derangement([0, 1, 2], rows, 7)

    def test_sr_uses_lr_inputs_hr_only_target(self):
        with tempfile.TemporaryDirectory() as directory:
            data_run, run, pseudo_path, rows = make_fixture(Path(directory))
            corpus = training.SlotCorpus(data_run, run, "fixture", "sr", "slots", "cpu", pseudo_path=pseudo_path)
            before = corpus.batch([0])
            expected = np.load(run / "fixture" / "lr_slots.npy")[0]
            np.testing.assert_array_equal(before[1][0].numpy(), expected)
            corpus.images = np.zeros_like(corpus.images)
            after = corpus.batch([0])
            torch.testing.assert_close(before[0], after[0])
            torch.testing.assert_close(before[1], after[1])
            self.assertFalse(torch.equal(before[2], after[2]))

    def test_baseline_needs_no_vlm_cache_and_invalid_cache_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            data_run, run, pseudo_path, rows = make_fixture(Path(directory))
            baseline = training.SlotCorpus(data_run, run, "absent", "segmentation", "image_only", "cpu",
                                          pseudo_path=pseudo_path)
            self.assertEqual(float(baseline.batch([0])[1].abs().sum()), 0.)
            np.save(run / "fixture" / "slots_done.npy", np.zeros((len(rows), 2), dtype=bool))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                training.SlotCorpus(data_run, run, "fixture", "segmentation", "slots", "cpu", pseudo_path=pseudo_path)

    def test_shuffle_stays_in_each_split(self):
        with tempfile.TemporaryDirectory() as directory:
            data_run, run, pseudo_path, rows = make_fixture(Path(directory))
            corpus = training.SlotCorpus(data_run, run, "fixture", "segmentation", "shuffled_slots", "cpu",
                                         pseudo_path=pseudo_path)
            for i, donor in corpus.donors.items():
                self.assertEqual(rows[i]["split"], rows[donor]["split"])
                self.assertNotEqual(rows[i]["subject_id"], rows[donor]["subject_id"])

    def test_rng_restore(self):
        saved = training.capture_rng()
        expected = (random.random(), np.random.random(), torch.rand(3))
        training.restore_rng(saved)
        actual = (random.random(), np.random.random(), torch.rand(3))
        self.assertEqual(expected[:2], actual[:2])
        torch.testing.assert_close(expected[2], actual[2], rtol=0, atol=0)

    def test_sr_objective_is_valid_pixel_mse_and_accumulates_per_sample(self):
        prediction = torch.tensor([[[[2., 100.]]], [[[3., 4.]]]])
        target = torch.zeros_like(prediction)
        mask = torch.tensor([[[[1., 0.]]], [[[1., 1.]]]])
        full = training.objective("sr", prediction, target, mask)
        self.assertAlmostEqual(float(full), (4 + (9 + 16) / 2) / 2)
        split = sum(training.objective("sr", prediction[i:i+1], target[i:i+1], mask[i:i+1]) for i in range(2)) / 2
        torch.testing.assert_close(full, split)

    def test_epoch_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_run, run, pseudo_path, rows = make_fixture(root)
            args = argparse.Namespace(data_run=data_run, run=run, model="fixture", task="segmentation",
                                      condition="slots", out=root / "uninterrupted", epochs=2, train_limit=None,
                                      batch_size=2, microbatch=1, learning_rate=3e-4, seed=7, device="cpu",
                                      threads=2, pseudo_path=pseudo_path)
            training.train(args)
            uninterrupted = torch.load(args.out / "checkpoint.pt", weights_only=False)
            args.out = root / "resumed"
            validate = training.validation_loss
            calls = [0]
            def fail_second_validation(*values):
                calls[0] += 1
                if calls[0] == 2:
                    raise RuntimeError("simulated interruption")
                return validate(*values)
            with mock.patch.object(training, "validation_loss", side_effect=fail_second_validation):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    training.train(args)
            training.train(args)
            resumed = torch.load(args.out / "checkpoint.pt", weights_only=False)
            self.assertEqual(resumed["epoch"], 2)
            self.assertEqual(resumed["step"], uninterrupted["step"])
            for key in uninterrupted["model"]:
                torch.testing.assert_close(uninterrupted["model"][key], resumed["model"][key], rtol=0, atol=0)
            metrics = json.loads((args.out / "metrics.json").read_text())
            self.assertEqual(metrics["metrics"]["human_test"]["n"], 2)
            self.assertEqual(metrics["train_n"], 2)

    def test_step_deadline_checkpoint_resumes_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_run, run, pseudo_path, rows = make_fixture(root)
            args = argparse.Namespace(data_run=data_run, run=run, model="fixture", task="segmentation",
                                      condition="slots", out=root / "uninterrupted", epochs=2, train_limit=None,
                                      batch_size=1, microbatch=1, learning_rate=3e-4, seed=7, device="cpu",
                                      threads=2, pseudo_path=pseudo_path, checkpoint_seconds=120.)
            training.train(args)
            uninterrupted = torch.load(args.out / "checkpoint.pt", weights_only=False)
            args.out = root / "deadline_resumed"
            # Stop after the first optimizer update, half-way through epoch 1.
            with mock.patch.object(training.StopRequest, "requested", side_effect=[False, True]):
                with self.assertRaises(SystemExit) as raised:
                    training.train(args)
            self.assertEqual(raised.exception.code, 124)
            self.assertFalse((args.out / "metrics.json").exists())
            partial = json.loads((args.out / "partial_metrics.json").read_text())
            self.assertFalse(partial["complete_requested_epochs"])
            self.assertEqual(partial["epochs"], .5)
            saved = torch.load(args.out / "checkpoint.pt", weights_only=False)
            self.assertEqual(saved["resume_batch_start"], 1)
            training.train(args)
            resumed = torch.load(args.out / "checkpoint.pt", weights_only=False)
            self.assertEqual(resumed["step"], uninterrupted["step"])
            for key in uninterrupted["model"]:
                torch.testing.assert_close(uninterrupted["model"][key], resumed["model"][key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
