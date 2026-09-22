"""Prepared-data integrity, linked supervision and global leakage boundaries."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from medworld.config import load_config
from medworld.datasets.prepared import PreparedData
from medworld.datasets.protocol import _sha256
from medworld.datasets.unified import UnifiedData
from medworld.downstream_tasks.registry import FINDINGS, MANUAL_SEGMENTATION_LAYOUT


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class PreparedDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / "prepared"
        self.root.mkdir()
        original = self.base / "original"
        original.mkdir()
        Image.new("L", (12, 16), color=100).save(original / "image.png")
        Image.new("L", (12, 16), color=255).save(original / "mask.png")
        (self.root / "images").symlink_to(original, target_is_directory=True)
        self.probabilities = original / "probs.npy"
        np.save(self.probabilities, np.full((2, 3, 256, 256), .75, dtype=np.float16))
        (self.root / "targets.npy").symlink_to(self.probabilities)
        self.classification = [
            {"id": f"class-{patient}", "subject_id": patient, "study_id": f"study-{patient}",
             "split": split, "image": "images/image.png", "box": [0, 64, 512, 384],
             "labels": [1, 0, -1, -2] + [0] * 9}
            for patient, split in (("train", "train"), ("held", "train"), ("test", "test"))]
        self.segmentation = [
            {"id": "pseudo-train", "subject_id": "train", "split": "train", "kind": "cxas",
             "image": "images/image.png", "box": [0, 64, 512, 384],
             "target_file": "targets.npy", "target_index": 0},
            {"id": "pseudo-test", "subject_id": "test", "split": "test", "kind": "cxas",
             "image": "images/image.png", "box": [0, 64, 512, 384],
             "target_file": "targets.npy", "target_index": 1},
            {"id": "human", "subject_id": "montgomery:1", "split": "human_test", "kind": "montgomery",
             "image": "images/image.png", "box": [0, 64, 512, 384],
             "masks": ["images/mask.png", "images/mask.png"]},
        ]
        vqa = self.root / "vqa"
        vqa.mkdir()
        for split in ("train", "valid", "test"):
            rows = ([{"idx": 1, "subject_id": "held", "image_path": "image.png", "question": "Edema?",
                      "answer": ["yes"], "semantic_type": "verify"}] if split == "test" else [])
            (vqa / f"{split}.json").write_text(json.dumps(rows))
        temporal = self.root / "temporal"
        temporal.mkdir()
        observations = [{"id": f"{patient}-{side}", "patient": patient, "split": "train",
                         "image": "../images/image.png", "report": f"Report {side}", "labels": [0] * 13}
                        for patient in ("train", "held") for side in ("before", "after")]
        write_rows(temporal / "observations.jsonl", observations)
        pairs = [{"id": f"pair-{patient}", "patient": patient, "split": "train",
                  "source": f"{patient}-before", "target": f"{patient}-after", "realized_gap_hours": 48.5}
                 for patient in ("train", "held")]
        for split in ("train", "validate", "test"):
            write_rows(temporal / f"{split}.jsonl", pairs if split == "train" else [])
        self.cfg = load_config(overrides={"prepared_data": str(self.root), "vqa_data": str(vqa),
                                         "image_root": str(self.root / "images"), "temporal_data": str(temporal)})
        self.write_manifest()

    def write_manifest(self):
        write_rows(self.root / "classification.jsonl", self.classification)
        write_rows(self.root / "segmentation.jsonl", self.segmentation)
        manifest = {"schema": "medworld-prepared-v1", "findings": list(FINDINGS),
                    "file_sha256": {name: _sha256(self.root / name)
                                    for name in ("classification.jsonl", "segmentation.jsonl",
                                                 "temporal/observations.jsonl", "temporal/train.jsonl",
                                                 "temporal/validate.jsonl", "temporal/test.jsonl")}}
        (self.root / "manifest.json").write_text(json.dumps(manifest))

    def test_linked_assets_supply_all_tasks_without_legacy_cohorts(self):
        data = PreparedData(self.cfg)
        classification = data.dataset("classification", "train")[0]
        self.assertEqual(classification["image"].size, (512, 512))
        self.assertEqual(classification["label_mask"][:4].tolist(), [True, True, False, False])
        self.assertFalse(any("report" in key for key in classification))
        pseudo = data.dataset("segmentation", "test")[0]
        torch.testing.assert_close(pseudo["targets"], torch.full((3, 256, 256), .75))
        self.assertEqual(pseudo["mask"].sum().item(), 256 * 192)
        human = data.dataset("segmentation", "human_test")[0]
        self.assertEqual(human["targets"].shape, (2, 256, 256))
        self.assertEqual(human["targets"].sum().item(), 2 * 256 * 192)
        vqa = data.dataset("vqa", "test")[0]
        self.assertEqual(vqa["question"], "Edema?")
        self.assertEqual(vqa["answer"], ["yes"])
        self.assertTrue((self.root / "targets.npy").is_symlink())
        self.assertEqual(list(self.root.glob("*.npy")), [self.root / "targets.npy"])

    def test_global_holdouts_remove_current_and_random_temporal_training_leakage(self):
        cfg = dict(self.cfg, decoded_image_cache=2)
        data = UnifiedData(cfg)
        self.assertEqual([r["subject_id"] for r in data.rows("classification", "train")], ["train"])
        self.assertEqual(len(data.rows("temporal", "train")), 2)
        forward, backward = data.rows("temporal", "train")
        self.assertEqual((forward["delta_hours"], backward["delta_hours"]), (48.5, -48.5))
        batch = data.batch("temporal", "train", [0])
        self.assertEqual(batch["labels"].shape, (1, 13))
        self.assertEqual(data.metadata["current_dropped_rows"]["classification/train"], 1)
        self.assertEqual(data.metadata["temporal_excluded_patients"]["train"], 1)
        self.assertEqual(data.batch("segmentation", "human_test", [0])["targets"].shape, (1, 2, 256, 256))

    def test_manifest_tampering_and_label_order_are_rejected(self):
        with (self.root / "classification.jsonl").open("a") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            PreparedData(self.cfg)
        self.write_manifest()
        path = self.root / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["findings"] = list(reversed(FINDINGS))
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "finding order"):
            PreparedData(self.cfg)

    def test_invalid_supervision_and_broken_links_fail_early(self):
        cases = [("target_index", 2, "index"), ("kind", "unknown", "kind"),
                 ("box", [0, 63, 512, 384], "ROI"), ("image", "missing.png", "missing")]
        for field, value, expected in cases:
            original = self.segmentation[0][field]
            with self.subTest(field=field):
                self.segmentation[0][field] = value
                self.write_manifest()
                with self.assertRaisesRegex((ValueError, FileNotFoundError), expected):
                    PreparedData(self.cfg)
            self.segmentation[0][field] = original
        self.write_manifest()
        np.save(self.probabilities, np.zeros((2, 2, 256, 256), dtype=np.float16))
        with self.assertRaisesRegex(ValueError, "shape"):
            PreparedData(self.cfg)

    def test_temporal_tampering_and_missing_fingerprints_are_rejected(self):
        path = self.root / "temporal/train.jsonl"
        with path.open("a") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "fingerprint.*temporal/train"):
            UnifiedData(self.cfg)
        self.write_manifest()
        path = self.root / "manifest.json"
        manifest = json.loads(path.read_text())
        del manifest["file_sha256"]["temporal/observations.jsonl"]
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "fingerprint.*temporal/observations"):
            UnifiedData(self.cfg)

    def test_prepared_data_rejects_historical_or_unrelated_temporal_root(self):
        with self.assertRaisesRegex(ValueError, "prepared_data/temporal"):
            UnifiedData(dict(self.cfg, temporal_data=str(self.base / "historical")))

    def test_source_path_changes_and_target_replacement_change_fingerprint(self):
        before = UnifiedData(self.cfg).fingerprint
        np.save(self.probabilities, np.full((2, 3, 256, 256), .25, dtype=np.float16))
        self.assertNotEqual(before, UnifiedData(self.cfg).fingerprint)

    def test_masks_are_only_arrays_opened(self):
        original_load = np.load
        loaded = []
        def checked(path, **kwargs):
            loaded.append(Path(path))
            return original_load(path, **kwargs)
        with patch("numpy.load", side_effect=checked):
            data = PreparedData(self.cfg)
            data.dataset("classification", "train")[0]
            data.dataset("segmentation", "test")[0]
        self.assertEqual(loaded, [self.root / "targets.npy"])

    def test_prepared_configuration_does_not_change_legacy_default(self):
        self.assertEqual(load_config()["prepared_data"], "")
        with self.assertRaisesRegex(ValueError, "prepared_data"):
            load_config(overrides={"prepared_data": None})


class ManualPreparedDataTests(unittest.TestCase):
    def setUp(self):
        self._manual = False
        PreparedDataTests.setUp(self)
        self._manual = True
        self.cfg.update(segmentation_channels=6, segmentation_sampling="balanced_dataset")
        self.segmentation = [{
            "id": "manual-cxr", "subject_id": "train", "split": "train", "kind": "human_cxr",
            "dataset": "mimic_cxr_human", "annotation": "expert_reviewed", "annotation_source": "human_reviewed",
            "image": "images/image.png", "box": [0, 64, 512, 384], "masks": ["images/mask.png"] * 2,
            "channels": [0, 1], "target_names": ["lungs", "heart"],
        }, {
            "id": "montgomery", "subject_id": "montgomery:1", "split": "human_test", "kind": "montgomery",
            "dataset": "montgomery", "annotation": "human", "image": "images/image.png",
            "box": [0, 64, 512, 384], "masks": ["images/mask.png"] * 2, "channels": [0], "target_names": ["lungs"],
        }]
        (self.root / "image.nii.gz").write_bytes(b"MRI decoder tested separately")
        (self.root / "mask.nii.gz").write_bytes(b"MRI decoder tested separately")
        for dataset, count in (("ucsf_alptdg", 3), ("mu_glioma_post", 1)):
            for index in range(count):
                self.segmentation.append({
                    "id": f"{dataset}:slice{index}", "subject_id": dataset, "volume_id": f"{dataset}:volume",
                    "split": "train", "kind": "mri", "dataset": dataset, "annotation": "expert_reviewed",
                    "image": "image.nii.gz", "mask_file": "mask.nii.gz", "box": [0, 64, 512, 384],
                    "t1ce_sha256": _sha256(self.root / "image.nii.gz"),
                    "mask_sha256": _sha256(self.root / "mask.nii.gz"),
                    "canonical_shape": [8, 6, 3], "slice_index": index, "normalization": [1., 99.],
                    "channels": [2, 3, 4, 5], "target_names": ["NETC", "SNFH", "ET", "RC"],
                })
        self.write_manifest()

    def write_manifest(self):
        PreparedDataTests.write_manifest(self)
        if self._manual:
            path = self.root / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest.update(schema="medworld-prepared-v2", segmentation_layout=list(MANUAL_SEGMENTATION_LAYOUT))
            path.write_text(json.dumps(manifest))

    def test_manual_cxr_masks_use_six_channels_and_montgomery_union(self):
        with patch("numpy.load", side_effect=AssertionError("Old pseudo arrays accessed")):
            data = PreparedData(self.cfg)
            current = data.dataset("segmentation", "train")[0]
            human = data.dataset("segmentation", "human_test")[0]
        self.assertEqual(current["targets"].shape, (6, 256, 256))
        self.assertEqual(current["mask"].shape, current["targets"].shape)
        self.assertEqual(current["active_channels"], [0, 1])
        self.assertEqual(current["targets"][2:].sum().item(), 0)
        self.assertEqual(current["mask"][2:].sum().item(), 0)
        self.assertEqual(current["mask"][:2].sum().item(), 2 * 256 * 192)
        self.assertEqual(human["targets"][0].sum().item(), 256 * 192)
        self.assertEqual(human["targets"][1:].sum().item(), 0)
        self.assertEqual(human["active_channels"], [0])
        self.assertEqual(data.segmentation_layout, list(MANUAL_SEGMENTATION_LAYOUT))

    def test_mixed_modality_collate_retains_dataset_volume_and_channel_semantics(self):
        data = PreparedData(self.cfg)
        targets = torch.zeros(6, 256, 256)
        targets[2:, :, 32:224] = 1
        with patch("medworld.datasets.mri_pixels.mri_source_canvas", return_value=torch.ones(1, 512, 512)), \
             patch("medworld.datasets.mri_pixels.mri_target", return_value=targets):
            examples = [data.dataset("segmentation", "train")[i] for i in (0, 1)]
        batch = data.collate("segmentation", examples)
        self.assertEqual(batch["targets"].shape, (2, 6, 256, 256))
        self.assertEqual(batch["mask"].shape, batch["targets"].shape)
        self.assertEqual(batch["segmentation_dataset"], ["mimic_cxr_human", "ucsf_alptdg"])
        self.assertEqual(batch["volume_id"], ["manual-cxr", "ucsf_alptdg:volume"])
        self.assertEqual(batch["active_channels"], [[0, 1], [2, 3, 4, 5]])
        self.assertEqual(batch["target_names"][1], ["NETC", "SNFH", "ET", "RC"])
        self.assertEqual(batch["mask"][1, :2].sum().item(), 0)

    def test_manual_schema_rejects_pseudo_and_mislabeled_targets(self):
        cases = [("kind", "cxas", "pseudo"), ("annotation", "pseudo", "annotation"),
                 ("annotation_source", "CXAS", "pseudo"), ("channels", [1, 0], "channels"),
                 ("target_names", ["heart", "lungs"], "target_names")]
        for field, value, message in cases:
            with self.subTest(field=field):
                original = self.segmentation[0][field]
                self.segmentation[0][field] = value
                self.write_manifest()
                with self.assertRaisesRegex(ValueError, message):
                    PreparedData(self.cfg)
                self.segmentation[0][field] = original
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "segmentation_channels=6"):
            PreparedData(dict(self.cfg, segmentation_channels=3))
        data = PreparedData(self.cfg)
        with self.assertRaisesRegex(ValueError, "pseudo-target"):
            data._array(str(self.probabilities))

    def test_manual_cxr_annotation_contents_affect_fingerprint(self):
        before = UnifiedData(self.cfg).fingerprint
        Image.new("L", (12, 16), color=0).save(self.root / "images/mask.png")
        self.assertNotEqual(before, UnifiedData(self.cfg).fingerprint)

    def test_balanced_dataset_stream_preserves_resume_and_distributed_partition(self):
        data = UnifiedData(self.cfg)
        # Inspect selected indices without decoding NIfTI fixtures. MRI pixel
        # geometry and labels are covered independently by mri_pixels tests.
        data.batch = lambda task, split, indices: list(indices)
        full = data.training_batch("segmentation", 0, 30, 42)
        names = [data.rows("segmentation", "train")[index]["dataset"] for index in full]
        self.assertEqual({name: names.count(name) for name in set(names)},
                         {"mimic_cxr_human": 10, "mu_glioma_post": 10, "ucsf_alptdg": 10})
        resumed = UnifiedData(self.cfg)
        resumed.batch = data.batch
        self.assertEqual(full[7:], resumed.training_batch("segmentation", 7, 23, 42))
        rank0 = data.training_batch("segmentation", 0, 15, 42)
        rank1 = data.training_batch("segmentation", 15, 15, 42)
        self.assertEqual(full, rank0 + rank1)
        ucsf_first_epoch = [full[index] for index in (2, 5, 8)]
        self.assertEqual(len(set(ucsf_first_epoch)), 3)
        uniform = UnifiedData(dict(self.cfg, segmentation_sampling="uniform"))
        self.assertNotEqual(data.fingerprint, uniform.fingerprint)


if __name__ == "__main__":
    unittest.main()
