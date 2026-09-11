import unittest

import torch

from mimic_vla_jepa.train_predictor import (
    TrainConfig,
    evaluation_batch_sums,
    finalize_evaluation_sums,
    interval_is_due,
    next_interval_seconds,
    training_should_stop,
    validate_resume_train_config,
    validate_train_config,
)


class WallClockScheduleTests(unittest.TestCase):
    def test_interval_boundaries_are_absolute_and_skip_missed_boundaries(self):
        next_due = next_interval_seconds(8.1 * 3600.0, 8.0)
        self.assertEqual(next_due, 16.0 * 3600.0)
        due, next_due = interval_is_due(33.0 * 3600.0, next_due, 8.0)
        self.assertTrue(due)
        self.assertEqual(next_due, 40.0 * 3600.0)
        due, next_due = interval_is_due(39.0 * 3600.0, next_due, 8.0)
        self.assertFalse(due)
        self.assertEqual(next_due, 40.0 * 3600.0)

    def test_either_training_limit_stops_the_run(self):
        self.assertTrue(
            training_should_stop(
                step=100,
                elapsed_seconds=1.0,
                max_steps=100,
                max_duration_hours=24.0,
            )
        )
        self.assertTrue(
            training_should_stop(
                step=1,
                elapsed_seconds=24.0 * 3600.0,
                max_steps=None,
                max_duration_hours=24.0,
            )
        )
        self.assertFalse(
            training_should_stop(
                step=99,
                elapsed_seconds=23.0 * 3600.0,
                max_steps=100,
                max_duration_hours=24.0,
            )
        )

    def test_unbounded_config_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            validate_train_config(TrainConfig(max_steps=None))
        with self.assertRaisesRegex(ValueError, "max_duration_hours"):
            validate_train_config(
                TrainConfig(max_steps=None, max_duration_hours=float("nan"))
            )

    def test_legacy_v2_train_config_can_resume_with_wall_clock_fields(self):
        legacy = {
            "seed": 42,
            "batch_size": 1,
            "max_steps": 5,
            "learning_rate": 1.0e-4,
            "weight_decay": 1.0e-8,
            "betas": (0.9, 0.95),
            "grad_clip": 1.0,
            "log_every": 1,
            "checkpoint_every": 0,
        }
        validate_resume_train_config(
            legacy,
            TrainConfig(
                max_steps=None,
                max_duration_hours=24.0,
                checkpoint_interval_hours=8.0,
                eval_interval_hours=8.0,
            ),
        )
        with self.assertRaisesRegex(ValueError, "learning_rate"):
            validate_resume_train_config(
                legacy,
                TrainConfig(learning_rate=2.0e-4),
            )


class EvaluationAggregationTests(unittest.TestCase):
    def test_batch_sums_are_additive_and_finalize_expected_metrics(self):
        target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        prediction = target.clone()
        source = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        zero_prediction = torch.zeros_like(target)
        shuffled_prediction = source.clone()

        first = evaluation_batch_sums(
            prediction=prediction[:1],
            source=source[:1],
            target=target[:1],
            zero_prediction=zero_prediction[:1],
            shuffled_prediction=shuffled_prediction[:1],
        )
        second = evaluation_batch_sums(
            prediction=prediction[1:],
            source=source[1:],
            target=target[1:],
            zero_prediction=zero_prediction[1:],
            shuffled_prediction=shuffled_prediction[1:],
        )
        result = finalize_evaluation_sums(first + second)

        self.assertEqual(result["examples"], 2)
        self.assertAlmostEqual(result["metrics"]["prediction"]["l1"], 0.0)
        self.assertAlmostEqual(
            result["metrics"]["prediction"]["copy_normalized_gain"], 1.0
        )
        self.assertAlmostEqual(result["metrics"]["copy_state"]["l1"], 1.0)
        self.assertAlmostEqual(result["metrics"]["zero_query"]["l1"], 0.5)
        self.assertAlmostEqual(
            result["paired_query_effect"]["shuffled_minus_correct_l1"], 1.0
        )
        self.assertAlmostEqual(
            result["paired_query_effect"]["zero_minus_correct_l1"], 0.5
        )


if __name__ == "__main__":
    unittest.main()
