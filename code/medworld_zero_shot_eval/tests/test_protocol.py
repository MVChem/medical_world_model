import tempfile
import unittest
from pathlib import Path

from medworld_zero_shot_eval.common import atomic, write_rows
from medworld_zero_shot_eval.prepare import sample_rows
from medworld_zero_shot_eval.score import comparison, parse, score, summarize


class ProtocolTest(unittest.TestCase):
    def test_sample_has_exact_size_and_does_not_select_on_answers(self):
        data = [
            {
                "idx": i,
                "image_id": str(i // 4),
                "semantic_type": ["verify", "choose", "query"][i % 3],
                "content_type": ["presence", "anatomy"][i % 2],
                "answer": ["yes"],
            }
            for i in range(200)
        ]
        selected = sample_rows(data, 37, 42)
        changed = [dict(r, answer=[]) for r in data]
        self.assertEqual(len(selected), 37)
        self.assertEqual(
            [r["idx"] for r in selected],
            [r["idx"] for r in sample_rows(changed, 37, 42)],
        )
        self.assertEqual(len({r["idx"] for r in selected}), 37)

    def test_empty_output_is_not_empty_answer(self):
        ref = {
            "id": "a",
            "patient": "p",
            "answer": [],
            "semantic_type": "query",
            "content_type": "attribute",
            "regional": False,
        }
        for response in (
            None,
            {"ok": True, "text": "", "finish_reason": "stop"},
            {"ok": True, "text": "[]", "finish_reason": "length"},
        ):
            row = comparison(ref, response, ["yes", "no"])
            self.assertEqual(row["exact"], 0)
            self.assertEqual(row["fp"], 1)
        row = comparison(
            ref, {"ok": True, "text": "[]", "finish_reason": "stop"}, ["yes", "no"]
        )
        self.assertEqual(row["exact"], 1)

    def test_multilabel_counts_and_format(self):
        ref = {
            "id": "a",
            "patient": "p",
            "answer": ["a", "b"],
            "semantic_type": "query",
            "content_type": "attribute",
            "regional": True,
        }
        row = comparison(
            ref,
            {"ok": True, "text": '["A", "c", "c"]', "finish_reason": "stop"},
            ["a", "b", "c"],
        )
        self.assertEqual((row["tp"], row["fp"], row["fn"]), (1, 1, 1))
        self.assertEqual(summarize([row])["micro_f1"], 0.5)
        self.assertEqual(parse('```json\n["A"]\n```', ["a"]), ({"a"}, None))
        self.assertIsNotNone(parse('["unknown"]', ["a"])[1])
        self.assertIsNotNone(parse("The answer is yes.", ["yes"])[1])

    def test_bootstrap_is_patient_clustered(self):
        records = [
            {
                "patient": "only_patient",
                "exact": 1,
                "tp": 1,
                "fp": 0,
                "fn": 0,
                "error": None,
                "empty_target": False,
            }
        ] * 3
        result = summarize(records, 100)
        self.assertEqual(result["patient_bootstrap_95ci"]["exact_match"], [1, 1])
        self.assertEqual(result["patient_bootstrap_95ci"]["micro_f1"], [1, 1])

    def test_diagnosis_excludes_training_patients_and_nonclinical_questions(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            refs = [
                {
                    "id": str(i),
                    "patient": str(i),
                    "answer": ["yes"],
                    "semantic_type": "verify",
                    "content_type": content,
                    "regional": False,
                }
                for i, content in enumerate(["presence", "presence", "plane", "gender"])
            ]
            write_rows(run / "references.jsonl", refs)
            atomic(run / "vocabulary.json", ["yes", "no"])
            atomic(run / "protocol.json", {})
            atomic(run / "training_overlap.json", {"overlap_sample_ids": ["0"]})
            write_rows(
                run / "model/predictions.jsonl",
                [
                    {
                        "id": r["id"],
                        "ok": True,
                        "text": '["yes"]',
                        "finish_reason": "stop",
                    }
                    for r in refs
                ],
            )
            result = score(run, "model")
            self.assertTrue(result["complete"])
            self.assertEqual(result["overall"]["n"], 4)
            self.assertEqual(result["diagnosis_all_sampled"]["n"], 2)
            self.assertEqual(result["diagnosis"]["n"], 1)
            self.assertEqual(result["vqa_train_disjoint"]["n"], 3)


if __name__ == "__main__":
    unittest.main()
