import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from PIL import Image
from medworld.datasets.current import MultiTaskData
from medworld.downstream_tasks.text.metrics import vqa_metrics


class VQATests(unittest.TestCase):
    def test_question_and_image_are_inputs_but_no_report_or_target_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / 'image.png'
            Image.new('RGB', (64, 32)).save(image)
            row = {'image': str(image), 'id': 'x', 'subject_id': '1', 'question': 'Is there edema?',
                   'answer': ['yes'], 'semantic_type': 'verify', 'report': 'MUST NOT ENTER INPUTS'}
            data = MultiTaskData.__new__(MultiTaskData)
            example = data._example('vqa', 'test', row)
            batch = data.collate('vqa', [example])
            self.assertEqual(batch['questions'], ['Is there edema?'])
            self.assertEqual(batch['answers'], [['yes']])
            self.assertEqual(batch['images'][0].size, (512, 512))
            self.assertFalse(any('report' in key for key in batch))

    def test_invalid_outputs_never_get_empty_answer_credit(self):
        rows = [{'answer': [], 'prediction': 'not json', 'semantic_type': 'query'},
                {'answer': ['yes'], 'prediction': '["yes"]', 'semantic_type': 'verify'},
                {'answer': [], 'prediction': '[]', 'semantic_type': 'query'}]
        scores = vqa_metrics(rows)
        self.assertEqual(scores['invalid'], 1)
        self.assertAlmostEqual(scores['exact_match'], 2 / 3)
        self.assertEqual(scores['by_type']['verify']['exact_match'], 1)
