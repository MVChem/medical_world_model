"""Data-boundary checks; synthetic arrays never touch the clinical caches."""
import unittest

import numpy as np
import torch

try:
    from .data import MultiTaskData, _patient_audit
except ImportError:
    from data import MultiTaskData, _patient_audit


class DataBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.data = MultiTaskData.__new__(MultiTaskData)
        self.data._arrays = {
            "current": np.zeros((1, 512, 512), dtype=np.uint8),
            "dense_lr": np.full((1, 128, 128), 37, dtype=np.uint8),
            "dense_hr": np.full((1, 512, 512), 211, dtype=np.uint8),
            "human": np.ones((1, 2, 256, 256), dtype=np.uint8),
        }
        self.row = {"id": "synthetic", "subject_id": "synthetic_patient",
                    "image_index": 0, "index": 0, "box": [0, 0, 512, 512],
                    "kind": "mimic", "labels": [1, 0, -1, -2] + [0] * 9,
                    "report_target": "Synthetic target text."}

    def test_hr_target_cannot_change_sr_inputs(self):
        before = self.data._example("sr", "test", self.row)
        self.data._arrays["dense_hr"][:] = 9
        after = self.data._example("sr", "test", self.row)
        self.assertTrue(torch.equal(before["pixels"], after["pixels"]))
        self.assertTrue(np.array_equal(np.asarray(before["image"]), np.asarray(after["image"])))
        self.assertFalse(torch.equal(before["targets"], after["targets"]))
        self.assertEqual(after["image"].size, (128, 128))

    def test_report_is_only_an_explicit_target(self):
        report = self.data._example("report", "train", self.row)
        classification = self.data._example("classification", "train", self.row)
        changed = dict(self.row, report_target="A completely different target.")
        other_report = self.data._example("report", "train", changed)
        self.assertTrue(torch.equal(report["pixels"], other_report["pixels"]))
        self.assertEqual(set(report) - set(classification), {"report_target"})
        batch = self.data.collate("report", [report, other_report])
        self.assertEqual(batch["report_targets"], [self.row["report_target"], changed["report_target"]])
        self.assertEqual(classification["label_mask"][:4].tolist(), [True, True, False, False])

    def test_external_human_target_has_only_two_lungs(self):
        row = dict(self.row, kind="montgomery", human_index=0)
        example = self.data._example("segmentation", "human_test", row)
        self.assertEqual(tuple(example["targets"].shape), (2, 256, 256))
        self.assertEqual(tuple(example["mask"].shape), (1, 256, 256))

    def test_cross_task_split_leakage_is_rejected(self):
        records = {"classification": {"train": [{"subject_id": "one"}]},
                   "report": {"test": [{"subject_id": "one"}]}}
        with self.assertRaisesRegex(ValueError, "Cross-task patient split leakage"):
            _patient_audit(records)
        records["report"]["test"][0]["subject_id"] = "two"
        self.assertTrue(_patient_audit(records)["globally_patient_disjoint"])

    def test_missing_official_tasks_fail_explicitly(self):
        for task in ("vqa", "grounding"):
            with self.assertRaises(NotImplementedError):
                self.data.dataset(task, "test")


if __name__ == "__main__":
    unittest.main()
