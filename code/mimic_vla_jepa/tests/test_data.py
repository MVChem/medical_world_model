import json
import tempfile
import unittest
from pathlib import Path

from mimic_vla_jepa.data import build_forecasting_prompt, load_transition_records


class DataLeakageTests(unittest.TestCase):
    def sample_row(self) -> dict:
        return {
            "task_type": "forecasting",
            "transition_id": "mimiccxr_0123456789abcdef",
            "split": "train",
            "elapsed_hours": 24.0,
            "horizon_bin": "0-24h",
            "source_image": "/restricted/current.jpg",
            "target_image": "/restricted/future.jpg",
            "source_report": {
                "findings": "Current lungs are clear.",
                "impression": None,
            },
            "future_report_for_eval_only": {"findings": "FUTURE_SECRET_EDEMA"},
            "target_state_for_eval_only": {"Edema": "present"},
            "state_delta_for_eval_only": {"Edema": "changed_to_present"},
        }

    def test_prompt_is_built_from_current_fields_only(self):
        prompt = build_forecasting_prompt(self.sample_row())
        self.assertIn("Current lungs are clear", prompt)
        self.assertIn("Prediction horizon: 0-24h", prompt)
        self.assertNotIn("24.0 hours", prompt)
        self.assertNotIn("FUTURE_SECRET_EDEMA", prompt)
        self.assertNotIn("changed_to_present", prompt)

    def test_prompt_uses_only_coarse_horizon_with_shared_boundaries(self):
        cases = [
            (24.0, "0-24h"),
            (24.000001, "24-72h"),
            (72.0, "24-72h"),
            (72.000001, "3-7d"),
            (168.0, "3-7d"),
            (168.000001, ">7d"),
        ]
        for elapsed_hours, expected_bin in cases:
            with self.subTest(elapsed_hours=elapsed_hours):
                row = self.sample_row()
                row["elapsed_hours"] = elapsed_hours
                row["horizon_bin"] = expected_bin
                prompt = build_forecasting_prompt(row)
                self.assertIn(f"Prediction horizon: {expected_bin}", prompt)
                self.assertNotIn(f"{elapsed_hours:.6f}", prompt)

        row_30 = self.sample_row()
        row_30.update(elapsed_hours=30.0, horizon_bin="24-72h")
        row_60 = self.sample_row()
        row_60.update(elapsed_hours=60.0, horizon_bin="24-72h")
        self.assertEqual(
            build_forecasting_prompt(row_30), build_forecasting_prompt(row_60)
        )

    def test_rejects_inconsistent_or_unknown_horizon_bin(self):
        row = self.sample_row()
        row["horizon_bin"] = "24-72h"
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            build_forecasting_prompt(row)
        row["horizon_bin"] = "FUTURE_SECRET_BIN"
        with self.assertRaisesRegex(ValueError, "horizon_bin"):
            build_forecasting_prompt(row)

    def test_legacy_manifest_derives_coarse_bin_without_exact_time(self):
        row = self.sample_row()
        del row["horizon_bin"]
        row["elapsed_hours"] = 60.0
        prompt = build_forecasting_prompt(row)
        self.assertIn("Prediction horizon: 24-72h", prompt)
        self.assertNotIn("60.0", prompt)

    def test_loader_does_not_reuse_combined_manifest_prompt(self):
        row = self.sample_row()
        row["prompt"] = "MALICIOUS_FUTURE_SECRET"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            records = load_transition_records(path, require_files=False)
        self.assertEqual(len(records), 1)
        self.assertNotIn("MALICIOUS_FUTURE_SECRET", records[0].prompt)

    def test_rejects_non_forecasting_task(self):
        row = self.sample_row()
        row["task_type"] = "controllable_generation"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "forecasting"):
                load_transition_records(path, require_files=False)


if __name__ == "__main__":
    unittest.main()
