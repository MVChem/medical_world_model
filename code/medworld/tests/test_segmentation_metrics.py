import unittest
from pathlib import Path
import torch
from medworld.downstream_tasks.segmentation.loss import segmentation_loss
from medworld.downstream_tasks.segmentation.metrics import segmentation_metrics, aggregate_segmentation
from medworld.config import load_config
from medworld.run_experiment import evaluation_jobs


class SegmentationMetricsTests(unittest.TestCase):
    def score(self, pred, target, mask=None):
        target = torch.tensor(target, dtype=torch.float32).reshape(1, 1, 2, 2)
        logits = torch.tensor(pred, dtype=torch.float32).reshape(1, 1, 2, 2) * 20 - 10
        mask = torch.ones_like(target) if mask is None else torch.tensor(mask).reshape_as(target)
        return segmentation_metrics(logits, target, mask)

    def test_partial_overlap_uses_union_not_dice_denominator(self):
        m = self.score([1, 1, 0, 0], [1, 0, 1, 0])
        self.assertAlmostEqual(m['mean_iou'], 1 / 3, places=6)
        self.assertAlmostEqual(m['mean_dice'], .5, places=6)

    def test_perfect_disjoint_empty_and_masked(self):
        for p, t, expected in [([1,0,0,0], [1,0,0,0], 1),
                               ([1,0,0,0], [0,1,0,0], 0),
                               ([0,0,0,0], [0,0,0,0], 1)]:
            self.assertAlmostEqual(self.score(p, t)['mean_iou'], expected, places=6)
        self.assertEqual(self.score([1,1,0,0], [1,0,0,0], [1,0,1,1])['mean_iou'], 1)

    def test_organ_average_and_reviewed_channel_selection(self):
        target = torch.zeros(1, 6, 2, 2)
        target[:, :2] = torch.tensor([1, 1, 0, 0, 1, 0, 1, 0]).reshape(1, 2, 2, 2)
        pred = torch.ones_like(target)
        pred[:, :2] = torch.tensor([1, 1, 0, 0, 1, 1, 0, 0]).reshape(1, 2, 2, 2)
        mask = torch.zeros_like(target)
        mask[:, :2] = 1
        m = segmentation_metrics(pred * 20 - 10, target, mask)
        self.assertEqual(len(m['iou_per_organ']), 2)
        self.assertAlmostEqual(m['mean_iou'], (1 + 1 / 3) / 2, places=6)
        self.assertAlmostEqual(m['mean_dice'], .75, places=6)

    def test_mismatched_channels_are_rejected_without_truncation_or_broadcast(self):
        target = torch.ones(1, 6, 2, 2)
        for score in (segmentation_metrics, segmentation_loss):
            for pred, mask in ((target[:, :3], target), (target, target[:, :1])):
                with self.subTest(score=score.__name__, pred=pred.shape, mask=mask.shape):
                    with self.assertRaisesRegex(ValueError, 'must align'):
                        score(pred, target, mask)

    def test_aggregation_requires_dataset_and_volume_metadata(self):
        row = {'id': 'slice', 'patient': 'patient', 'dataset': 'ucsf_alptdg',
               'volume_id': 'volume', 'target_names': ['NETC']}
        for key in ('dataset', 'volume_id', 'target_names'):
            incomplete = {k: v for k, v in row.items() if k != key}
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'records require'):
                aggregate_segmentation([incomplete])

    def test_experiment_config_keeps_both_segmentation_tests(self):
        cfg = load_config(Path(__file__).resolve().parents[1] / 'configs/medworld_0922.json')
        jobs = {j['id'] for j in evaluation_jobs(Path('/run'), cfg)}
        self.assertTrue({'segmentation', 'segmentation_human'} <= jobs)
