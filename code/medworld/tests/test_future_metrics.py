import math
import unittest
from unittest.mock import Mock, patch

from medworld.evaluation.future_metrics import (
    DIRECTIONS, SCHEMA, OfficialRadGraphScorer, reference_fingerprint,
    score_future_task, validate_radgraph_provenance,
)


def refs(targets, **extra):
    return [{"id": str(i), "patient": str(i // 2), "target": target, **extra}
            for i, target in enumerate(targets)]


def protocol(task, rows, **extra):
    return {"schema": SCHEMA, "task": task,
            "unit": "days" if task == "remaining_los" else "fraction",
            "reference_sha256": reference_fingerprint(rows), **extra}


def provenance():
    return {"package_version": "0.1.18", "model_type": "radgraph-xl",
            "reward_level": "partial", "report_scope": "full_report",
            "model_files_sha256": {key: "0" * 64 for key in ("weights.th", "config.json")},
            "scorer_files_sha256": {key: "0" * 64 for key in ("radgraph.py", "rewards.py", "utils.py")}}


class FutureMetricTests(unittest.TestCase):
    def test_categorical_parse_error_counts_wrong_but_missing_record_incomplete(self):
        rows = refs(["Yes", "no", "yes"])
        spec = protocol("future_vqa", rows, vocabulary=["yes", "no"])
        result = score_future_task("future_vqa", rows, {"0": " YES ", "1": ["no"], "2": "yes"}, protocol=spec)
        self.assertTrue(result["complete"])
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertEqual(result["invalid_ids"], ["1"])
        missing = score_future_task("future_vqa", rows, {"0": "yes", "1": "no"}, protocol=spec)
        self.assertFalse(missing["complete"])
        self.assertIsNone(missing["accuracy"])
        self.assertEqual(missing["n"], 3)
        self.assertEqual(missing["missing_ids"], ["2"])

    def test_frozen_cohort_rejects_changed_reference_extra_ids_and_duplicates(self):
        rows = refs([0, 1])
        spec = protocol("mortality_30d", rows, horizon_days=30)
        for changed, predicted in ((refs([1, 1]), {"0": 0.2, "1": 0.8}),
                                   (rows, {"0": 0.2, "1": 0.8, "other": 0.1}),
                                   ([rows[0], rows[0]], {"0": 0.2})):
            with self.subTest(changed=changed, predicted=predicted), self.assertRaises(ValueError):
                score_future_task("mortality_30d", changed, predicted, protocol=spec)
        self.assertEqual(reference_fingerprint(rows), reference_fingerprint(list(reversed(rows))))

    def test_progression_balances_three_classes_not_findings_or_accuracy(self):
        rows = refs(list(DIRECTIONS), finding="edema")
        other = [{"id": f"b{i}", "patient": "other", "finding": "effusion", "target": value}
                 for i, value in enumerate(["improved", "stable"] + ["worsened"] * 8)]
        rows += other
        predictions = {row["id"]: "improved" for row in rows}
        predictions.update({row["id"]: row["target"] for row in other})
        predictions["2"] = None  # A present parse error is still an evaluated answer.
        result = score_future_task("progression", rows, predictions,
                                   protocol=protocol("progression", rows, findings=["edema", "effusion"]))
        self.assertTrue(result["complete"])
        self.assertAlmostEqual(result["balanced_accuracy"], (1 + 1 / 2 + 8 / 9) / 3)
        self.assertEqual(result["per_finding"]["effusion"]["support"]["worsened"], 8)
        self.assertEqual(result["invalid"], 1)

    def test_progression_sparse_findings_do_not_invalidate_three_class_cohort(self):
        rows = [{"id": str(i), "patient": "p", "finding": f"finding{i}", "target": direction}
                for i, direction in enumerate(DIRECTIONS)]
        result = score_future_task("progression", rows, {r["id"]: r["target"] for r in rows},
                                   protocol=protocol("progression", rows, findings=[r["finding"] for r in rows]))
        self.assertTrue(result["complete"])
        self.assertEqual(result["balanced_accuracy"], 1.0)
        self.assertEqual(len(result["unsupported_findings"]), 3)

    def test_progression_never_changes_three_class_support_rule(self):
        rows = refs(["stable"], finding="edema")
        result = score_future_task("progression", rows, {"0": "stable"},
                                   protocol=protocol("progression", rows, findings=["edema"]))
        self.assertFalse(result["complete"])
        self.assertIsNone(result["balanced_accuracy"])
        self.assertEqual(result["unsupported_findings"], ["edema"])
        rows = refs(["unmentioned"], finding="edema")
        with self.assertRaisesRegex(ValueError, "assessable"):
            score_future_task("progression", rows, {"0": "stable"},
                              protocol=protocol("progression", rows, findings=["edema"]))

    def test_numeric_metrics_ties_units_missing_and_nonfinite(self):
        rows = refs([0, 1, 1, 0])
        spec = protocol("mortality_30d", rows, horizon_days=30)
        result = score_future_task("mortality_30d", rows, dict.fromkeys(["0", "1", "2", "3"], 0.5), protocol=spec)
        self.assertEqual(result["auroc"], 0.5)
        self.assertEqual(result["support"], {"negative": 2, "positive": 2})
        for bad in (None, float("nan"), float("inf"), -0.1, 1.1, True, "0.5"):
            predictions = dict.fromkeys(["0", "1", "2", "3"], 0.5)
            predictions["0"] = bad
            with self.subTest(bad=bad):
                result = score_future_task("mortality_30d", rows, predictions, protocol=spec)
                self.assertFalse(result["complete"])
                self.assertIsNone(result["auroc"])
                self.assertEqual(result["n"], 4)
        los = refs([2, 7.5])
        result = score_future_task("remaining_los", los, {"0": 1, "1": 5},
                                  protocol=protocol("remaining_los", los))
        self.assertEqual(result["mae_days"], 1.75)
        self.assertEqual(result["unit"], "days")
        with self.assertRaisesRegex(ValueError, "unit"):
            score_future_task("remaining_los", los, {"0": 1, "1": 5},
                              protocol={**protocol("remaining_los", los), "unit": "hours"})
        with self.assertRaisesRegex(ValueError, "exactly 30"):
            score_future_task("mortality_30d", rows, {}, protocol={**spec, "horizon_days": 28})

    def test_one_class_mortality_is_unavailable_and_invalid_reference_rejected(self):
        rows = refs([1, 1])
        result = score_future_task("mortality_30d", rows, {"0": 0.9, "1": 0.8},
                                  protocol=protocol("mortality_30d", rows, horizon_days=30))
        self.assertFalse(result["complete"])
        self.assertIsNone(result["auroc"])
        rows = refs([-1])
        with self.assertRaisesRegex(ValueError, "reference"):
            score_future_task("remaining_los", rows, {"0": 1}, protocol=protocol("remaining_los", rows))

    def test_official_radgraph_adapter_counts_empty_report_zero_and_checks_output(self):
        rows = refs(["No edema.", "Edema."])
        spec = protocol("future_report", rows, radgraph=provenance())
        scorer = object.__new__(OfficialRadGraphScorer)
        scorer.provenance = provenance()
        scorer._scorer = Mock(return_value=(0.8, [0.8], [], []))
        result = score_future_task("future_report", rows, {"0": "No edema.", "1": ""},
                                  protocol=spec, radgraph_scorer=scorer)
        self.assertTrue(result["complete"])
        self.assertEqual(result["radgraph_f1"], 0.4)
        self.assertEqual(result["invalid"], 1)
        scorer._scorer.assert_called_once_with(refs=["No edema."], hyps=["No edema."])
        scorer._scorer.return_value = (1, [math.nan], [], [])
        failed = score_future_task("future_report", rows, {"0": "No edema.", "1": ""},
                                   protocol=spec, radgraph_scorer=scorer)
        self.assertFalse(failed["complete"])
        self.assertIsNone(failed["radgraph_f1"])
        with self.assertRaisesRegex(ValueError, "no lexical fallback"):
            score_future_task("future_report", rows, {"0": "No edema.", "1": ""},
                              protocol=spec, radgraph_scorer=lambda *args: 1.0)

    def test_radgraph_requires_version_hashes_and_offline_before_import(self):
        for change in ({"reward_level": "simple"}, {"package_version": "latest"},
                       {"model_files_sha256": {}}, {"scorer_files_sha256": {"../escape": "0" * 64}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_radgraph_provenance({**provenance(), **change})
        with patch.dict("os.environ", {}, clear=True), patch("importlib.metadata.version") as version:
            with self.assertRaisesRegex(ValueError, "OFFLINE"):
                OfficialRadGraphScorer(provenance(), model_cache_dir="/missing", tokenizer_cache_dir="/missing")
            version.assert_not_called()


if __name__ == "__main__":
    unittest.main()
