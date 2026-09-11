from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mimic_vla_jepa.summarize_serious import summarize


def evaluation(step: int, hours: float, prediction_l1: float, phase: str | None):
    row = {
        "event": "evaluation",
        "step": step,
        "elapsed_seconds": hours * 3600.0,
        "examples": 10,
        "metrics": {
            "prediction": {"l1": prediction_l1, "cosine_distance": 0.5},
            "copy_state": {"l1": 1.0, "cosine_distance": 0.4},
            "zero_query": {"l1": 1.2, "cosine_distance": 0.6},
            "shuffled_query": {"l1": 1.3, "cosine_distance": 0.7},
        },
        "paired_query_effect": {
            "prediction_change_l1_zero": 0.2,
            "prediction_change_l1_shuffled": 0.3,
        },
    }
    if phase is not None:
        row["phase"] = phase
    return row


class SeriousSummaryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.run_dir = self.root / "run"
        self.checkpoint_dir = self.root / "checkpoints"
        self.run_dir.mkdir()
        self.checkpoint_dir.mkdir()
        (self.run_dir / "run.json").write_text(
            json.dumps(
                {
                    "config": {
                        "train": {
                            "expected_world_size": 4,
                            "max_duration_hours": 24.0,
                            "eval_interval_hours": 8.0,
                            "checkpoint_interval_hours": 8.0,
                        }
                    },
                    "checkpoint_dir": str(self.checkpoint_dir),
                    "train_examples": 20_000,
                    "eval_examples": 932,
                    "world_size": 4,
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_metrics(self, rows: list[dict]) -> None:
        (self.run_dir / "metrics.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    def test_partial_run_reports_trend_without_claiming_completion(self) -> None:
        self.write_metrics(
            [
                evaluation(0, 0.0, 1.5, "initial"),
                {
                    "event": "train",
                    "step": 10,
                    "elapsed_seconds": 200.0,
                    "loss_l1": 1.4,
                    "gradient_norm": 0.4,
                    "learning_rates": {"predictor": 1.0e-6},
                },
            ]
        )
        value = summarize(self.run_dir)
        self.assertFalse(value["audit"]["complete"])
        self.assertEqual(value["training"]["latest_step"], 10)
        self.assertEqual(value["evaluations"][0]["delta_l1_from_initial"], 0.0)

    def test_completed_run_requires_all_milestones_and_checkpoints(self) -> None:
        self.write_metrics(
            [
                evaluation(0, 0.0, 1.5, "initial"),
                evaluation(100, 8.0, 1.3, None),
                evaluation(200, 16.0, 1.1, None),
                evaluation(300, 24.0, 0.9, "final"),
            ]
        )
        (self.run_dir / "result.json").write_text(
            json.dumps({"elapsed_seconds": 24.0 * 3600.0}), encoding="utf-8"
        )
        (self.checkpoint_dir / "checkpoint_step-000100.pt").touch()
        (self.checkpoint_dir / "checkpoint_step-000200.pt").touch()
        (self.checkpoint_dir / "checkpoint_last.pt").touch()

        value = summarize(self.run_dir)
        self.assertTrue(value["audit"]["complete"])
        final = value["evaluations"][-1]
        self.assertAlmostEqual(final["relative_l1_improvement"], 0.4)
        self.assertAlmostEqual(final["copy_normalized_gain"], 0.1)


if __name__ == "__main__":
    unittest.main()
