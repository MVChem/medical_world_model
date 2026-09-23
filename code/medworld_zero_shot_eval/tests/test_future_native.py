import math
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from medworld.datasets.vqa import VOCABULARY
from medworld_zero_shot_eval.future_evaluate import (
    DIRECTIONS, LOS_DAYS, OPTION_CODES, future_prompt, native_readout_protocol,
    option_ids, option_probabilities, predict_future,
)


class Tokenizer:
    def encode(self, text, **kwargs):
        if text in ('Yes', 'No'):
            return [500 if text == 'Yes' else 501]
        return [OPTION_CODES.index(text)]

    def decode(self, ids, **kwargs):
        return 'éééABC'


class NoTargetBatch(dict):
    def __getitem__(self, key):
        if key in ('labels', 'answers', 'target', 'target_image', 'target_report'):
            raise AssertionError('A future target entered native prediction')
        return super().__getitem__(key)


class NativeFutureTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {'future_source_bytes': 383, 'future_report_tokens': 6,
                    'context_tokens': 384, 'vision_pixels': 256}
        self.batch = NoTargetBatch(images=[object()], reports=['Available source report'],
                                  delta_hours=torch.tensor([36.]), questions=['Will edema improve?'],
                                  labels=1, answers='secret target', target_report='future evidence')
        self.processor = SimpleNamespace(tokenizer=Tokenizer())

    @staticmethod
    def inputs(processor, image, prompt, pixels, device):
        return {'input_ids': torch.tensor([[10, 20]]), 'prompt': prompt}

    def model(self, scores):
        owner = self
        class Model:
            calls = 0

            def to(self, device):
                return self

            def eval(self):
                return self

            def requires_grad_(self, enabled):
                owner.assertFalse(enabled)
                return self

            def __call__(self, **kwargs):
                owner.assertEqual(kwargs['logits_to_keep'], 1)
                owner.assertFalse(kwargs['use_cache'])
                self.calls += 1
                return SimpleNamespace(logits=scores.reshape(1, 1, -1))

            def generate(self, **kwargs):
                owner.assertFalse(kwargs['do_sample'])
                owner.assertEqual(kwargs['max_new_tokens'], owner.cfg['future_report_tokens'])
                owner.assertTrue(bool(kwargs['stopping_criteria'](torch.tensor([[10, 20, 30]]), None)))
                return torch.tensor([[10, 20, 30]])

        return Model()

    def test_codebook_checks_unique_single_tokens(self):
        self.assertEqual(len(OPTION_CODES), len(VOCABULARY))
        self.assertEqual(option_ids(Tokenizer(), len(VOCABULARY)), list(range(110)))
        for response in ([1, 2], [1]):
            with self.subTest(response=response), self.assertRaisesRegex(ValueError, 'single-token'):
                option_ids(SimpleNamespace(encode=lambda *a, **kw: response), 3)

    def test_option_probabilities_ignore_unrelated_logits(self):
        scores = torch.tensor([1., 3., 1e6])
        result = option_probabilities(scores, [0, 1])
        self.assertAlmostEqual(float(result[1]), float(torch.sigmoid(torch.tensor(2.))), places=6)
        with self.assertRaisesRegex(ValueError, 'finite'):
            option_probabilities(torch.tensor([float('nan'), 1.]), [0, 1])

    def test_one_forward_vqa_and_progression_predictions_ignore_targets(self):
        for task, options, winner in [('future_vqa', VOCABULARY, 109), ('progression', DIRECTIONS, 2)]:
            logits = torch.zeros(1000)
            logits[winner], logits[-1] = 10., 1e6
            model = self.model(logits)
            with patch('medworld_zero_shot_eval.future_evaluate.native_inputs', side_effect=self.inputs):
                prediction = predict_future(model, self.processor, task, self.batch, self.cfg, 'cpu')
            self.assertEqual(prediction, options[winner])
            self.assertEqual(model.calls, 1)
            prompt, _ = future_prompt(task, self.batch, self.cfg)
            self.assertNotIn('secret target', prompt)
            self.assertNotIn('future evidence', prompt)

    def test_mortality_uses_continuous_yes_no_probability(self):
        self.batch['delta_hours'] = torch.tensor([720.])
        logits = torch.zeros(1000)
        logits[500] = math.log(3)
        model = self.model(logits)
        with patch('medworld_zero_shot_eval.future_evaluate.native_inputs', side_effect=self.inputs):
            probability = predict_future(model, self.processor, 'mortality_30d', self.batch, self.cfg, 'cpu')
        self.assertAlmostEqual(probability, .75, places=6)
        self.assertEqual(model.calls, 1)

    def test_los_has_fixed_horizon_and_target_independent_day_grid(self):
        self.batch['delta_hours'] = torch.tensor([24.])
        logits = torch.zeros(1000)
        model = self.model(logits)
        with patch('medworld_zero_shot_eval.future_evaluate.native_inputs', side_effect=self.inputs):
            prediction = predict_future(model, self.processor, 'remaining_los', self.batch, self.cfg, 'cpu')
        self.assertAlmostEqual(prediction, sum(LOS_DAYS) / len(LOS_DAYS), places=4)
        self.assertEqual(native_readout_protocol(self.cfg)['remaining_los']['options_days'], list(LOS_DAYS))
        self.batch['delta_hours'] = torch.tensor([72.])
        with self.assertRaisesRegex(ValueError, 'fixed horizon'):
            future_prompt('remaining_los', self.batch, self.cfg)

    def test_reports_and_questions_share_byte_caps_and_generation_uses_no_targets(self):
        self.batch['reports'] = ['S' * 383 + 'SECRET_SUFFIX']
        self.batch['questions'] = ['Q' * 383 + 'SECRET_QUESTION_SUFFIX']
        prompt, _ = future_prompt('future_vqa', self.batch, self.cfg)
        self.assertNotIn('SECRET_', prompt)
        with patch('medworld_zero_shot_eval.future_evaluate.native_inputs', side_effect=self.inputs):
            answer = predict_future(self.model(torch.zeros(1000)), self.processor, 'future_report',
                                    self.batch, self.cfg, 'cpu')
        self.assertEqual(answer, 'éé')
        self.assertLessEqual(len(answer.encode('utf-8')), self.cfg['future_report_tokens'] - 1)

    def test_cli_writes_shared_scorer_protocol_and_fingerprinted_reference(self):
        from medworld.config import load_config
        from medworld.evaluation.future_metrics import reference_fingerprint
        from medworld.evaluation.protocol import metric_protocol
        from medworld_zero_shot_eval.future_evaluate import main
        from medworld_zero_shot_eval.models import models
        qwen = next(row for row in models() if row['id'] == 'qwen9b')
        cfg = load_config(overrides={'future_enabled': True, 'qwen': qwen['path']})
        target = VOCABULARY[3]
        row = {'id': 'future_vqa:test:0', 'patient': 'p1', 'answer': target}

        def batch(task, split, indices, *, source_only):
            self.assertTrue(source_only)
            self.assertEqual((task, split, indices), ('future_vqa', 'test', [0]))
            return self.batch

        future = SimpleNamespace(rows=lambda task, split: [row], batch=batch, fingerprint='future-cohort')
        data = SimpleNamespace(future=future, fingerprint='data-cohort', metadata={'future': 'fixed'})
        logits = torch.zeros(1000)
        logits[3] = 10.
        model = self.model(logits)
        with TemporaryDirectory() as temporary:
            out = Path(temporary) / 'evaluation'
            with patch('sys.argv', ['future_evaluate', '--config', 'unused.json', '--out', str(out),
                                   '--tasks', 'future_vqa', '--limit', '1']), \
                 patch('medworld.config.load_config', return_value=cfg), \
                 patch('medworld.gpu.acquire_gpu', return_value=(None, 'cpu')), \
                 patch('medworld.datasets.UnifiedData', return_value=data), \
                 patch('transformers.AutoProcessor.from_pretrained', return_value=self.processor), \
                 patch('transformers.AutoModelForImageTextToText.from_pretrained', return_value=model), \
                 patch('medworld.datasets.protocol._sha256', return_value='a' * 64), \
                 patch('medworld_zero_shot_eval.future_evaluate.native_inputs', side_effect=self.inputs):
                main()
            summary = json.loads((out / 'summary.json').read_text())
            self.assertEqual(summary['metric_protocol'], metric_protocol('table1'))
            self.assertEqual(summary['tasks']['future_vqa']['accuracy'], 1.)
            reference = {'id': row['id'], 'patient': row['patient'], 'target': target}
            self.assertEqual(summary['references']['future_vqa']['references_sha256'], reference_fingerprint([reference]))
            self.assertTrue(json.loads((out / 'status.json').read_text())['partial'])
            prediction = json.loads((out / 'future_vqa.jsonl').read_text())
            self.assertEqual(prediction, {**reference, 'prediction': target})
            self.assertTrue((out / 'future_vqa_protocol.json').is_file())


if __name__ == '__main__':
    unittest.main()
