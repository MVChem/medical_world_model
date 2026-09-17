import unittest
import numpy as np
from medworld_zero_shot.common import prompt_for, require_complete, temporal_input
from medworld_zero_shot.common import CURRENT_FINDINGS
from stats import probability_metrics
from medworld.evaluate import classification_metrics


class ContractTests(unittest.TestCase):
    def test_target_and_ehr_cannot_enter_temporal_prompt(self):
        row = dict(direction="forward", delta_hours=27.125, source_observation_text="source only")
        before = prompt_for("table1_report", row)
        row.update(target_report="SECRET", target_image="SECRET", labels=[1]*6, ehr="SECRET", horizon=3)
        self.assertEqual(before, prompt_for("table1_report", row))
        self.assertIn("27.125 hours later", before)
        row.update(direction="backward", delta_hours=-27.125)
        self.assertIn("27.125 hours earlier", prompt_for("table1_report", row))
        row["delta_hours"] = 27.125
        with self.assertRaises(ValueError):
            prompt_for("table1_report", row)

    def test_current_report_withheld(self):
        row = dict(report="SECRET", labels=[1]*13, source_observation_text="SECRET")
        self.assertNotIn("SECRET", prompt_for("table2_report", row))
        self.assertNotIn("SECRET", prompt_for("table2_prob", row, "Edema"))

    def test_full_cohort_required(self):
        expected = [dict(id="a"), dict(id="b")]
        for bad in (expected[:1], expected[::-1], [expected[0], expected[0]]):
            with self.assertRaises(ValueError):
                require_complete(expected, bad)
        require_complete(expected, expected)

    def test_probability_masks_match_unified_evaluator(self):
        rng = np.random.default_rng(42)
        labels = rng.choice([-2, -1, 0, 1], (50, 13))
        probabilities = rng.random((50, 13))
        labels[:, 0] = -1
        labels[:, 1] = 1
        ours = classification_metrics(labels, probabilities, CURRENT_FINDINGS)
        baseline = probability_metrics(labels, probabilities, CURRENT_FINDINGS)
        self.assertEqual(ours["macro_ap"], baseline["ap"])
        self.assertEqual(ours["macro_auroc"], baseline["auroc"])
        probabilities[0, 0] = np.nan
        with self.assertRaises(ValueError):
            probability_metrics(labels, probabilities, CURRENT_FINDINGS)

    def test_source_text_includes_exact_encoder_prefix_in_budget(self):
        class Tokenizer:
            def __call__(self, text, truncation, max_length):
                return {"input_ids":[ord(c) for c in text][:max_length]}
            def decode(self, ids, skip_special_tokens):
                return "".join(chr(i) for i in ids)
        pair = dict(id="pair:forward", patient=1, direction="forward", delta_hours=12)
        source = dict(report="abc" * 200, target_report="SECRET")
        result = temporal_input(pair, source, Tokenizer(), 45)
        expected = ("Chest radiograph observation.\nReport:\n" + source["report"])[:45]
        self.assertEqual(result["source_observation_text"], expected)
        self.assertEqual(len(result["source_text_token_ids"]), 45)
        self.assertNotIn("SECRET", str(result))


if __name__ == "__main__":
    unittest.main()
