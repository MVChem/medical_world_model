import unittest
from typing import ClassVar

from medworld_vqa.literature_score import comparison, text_labels
from medworld_vqa.score import INVALID


class TextAdapterTest(unittest.TestCase):
    vocabulary: ClassVar[set[str]] = {
        "yes",
        "no",
        "pleural effusion",
        "pneumonia",
        "atelectasis",
        "m",
        "f",
    }

    def test_binary_boundary_and_explanation(self):
        self.assertEqual(
            text_labels("**Yes**, there is an opacity.", "verify", self.vocabulary),
            ({"yes"}, None),
        )
        self.assertEqual(
            text_labels("No pleural effusion is seen.", "verify", self.vocabulary),
            ({"no"}, None),
        )
        self.assertIn(
            INVALID, text_labels("Nothing definite.", "verify", self.vocabulary)[0]
        )

    def test_lists_keep_extra_answers_and_do_not_match_negated_substrings(self):
        self.assertEqual(
            text_labels(
                "Pneumonia, pleural effusion and atelectasis.", "query", self.vocabulary
            )[0],
            {"pneumonia", "pleural effusion", "atelectasis"},
        )
        self.assertEqual(
            text_labels("No pleural effusion.", "query", self.vocabulary)[0], {INVALID}
        )
        self.assertEqual(
            text_labels("Pleural effusion, unknown disease", "query", self.vocabulary)[
                0
            ],
            {"pleural effusion", INVALID},
        )
        self.assertEqual(text_labels("Female", "choose", self.vocabulary)[0], {"f"})

    def test_failed_and_blank_answers_do_not_receive_empty_credit(self):
        ref = {
            "id": "1",
            "patient": "p",
            "answer": [],
            "semantic_type": "query",
            "content_type": "attribute",
            "regional": False,
        }
        for pred in (
            None,
            {"ok": True, "text": "", "finish_reason": "stop"},
            {"ok": True, "text": "None", "finish_reason": "length"},
        ):
            self.assertEqual(comparison(ref, pred, self.vocabulary)["exact"], 0)
        self.assertEqual(
            comparison(
                ref,
                {"ok": True, "text": "None", "finish_reason": "stop"},
                self.vocabulary,
            )["exact"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
