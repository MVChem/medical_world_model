import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from medworld.gpu import ALLOWED_GPUS, acquire_gpu
from medworld_spatial.night import plan
from medworld_spatial.online import OnlineInputs, SourceData
from medworld_spatial.train import save_group, restore_group


class OnlineTests(unittest.TestCase):
    def test_online_teachers_never_read_hr_or_task_targets(self):
        online = OnlineInputs.__new__(OnlineInputs)
        online.device, online.views, online.selection_seed = 'cpu', 2, 42
        online.model = object()
        online.reader = {'merge': 2}
        online.projection = torch.randn(4, 64)
        online.verified_reader = True
        class Teacher:
            def scores(self, pixels):
                return pixels.mean((1, 2, 3))[:, None, None, None].expand(-1, 4, 2, 2)
        online.teacher = Teacher()
        def extract(_model, images, full):
            mean = torch.tensor([np.array(image).mean() / 255 for image in images]).float()
            raw = mean[:, None, None, None].expand(-1, 4, 4, 4).clone()
            slots = mean[:, None, None].expand(-1, 8 if full else 4, 1024).clone()
            return raw, slots, (2, 2)
        batch = {'pixels': torch.rand(2, 1, 128, 128), 'targets': torch.rand(2, 1, 512, 512),
                 'valid': torch.ones(2, 1, 512, 512), 'ids': ['a', 'b'], 'patients': ['1', '2']}
        with patch('medworld_spatial.online.extract', side_effect=extract), patch('torch.save') as write:
            first = online.features(batch)
            batch['targets'] = torch.zeros_like(batch['targets'])
            second = online.features(batch)
            for key in ('raw', 'fusion', 'teacher_features', 'thetas', 'semantic_probabilities'):
                torch.testing.assert_close(first[key], second[key], atol=0, rtol=0)
                self.assertFalse(first[key].requires_grad)
            write.assert_not_called()

    def test_stream_resume_and_seed_matching(self):
        source = SourceData.__new__(SourceData)
        source.rows = {('segmentation', 'train'): [None] * 7}
        source.permutations = {}
        first = source.training_indices('segmentation', 0, 12, 42)
        source.permutations = {}
        resumed = source.training_indices('segmentation', 8, 4, 42)
        self.assertEqual(first[8:], resumed)
        self.assertEqual(len(set(first[:7])), 7)
        self.assertNotEqual(first, source.training_indices('segmentation', 0, 12, 43))

    def test_group_checkpoint_restores_optimizer_trajectory_and_rejects_changed_inputs(self):
        torch.manual_seed(3)
        models = {v: torch.nn.Linear(3, 2) for v in ('a', 'b')}
        optimizers = {v: torch.optim.AdamW(m.parameters(), lr=.01) for v, m in models.items()}
        inputs = torch.randn(4, 3)
        def advance(ms, opts):
            for v, m in ms.items():
                opts[v].zero_grad()
                m(inputs).square().mean().backward()
                opts[v].step()
        advance(models, optimizers)
        protocol = {'source': 'unchanged', 'seed': 42}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'last.pt'
            save_group(path, models, optimizers, 1, protocol)
            restored = copy.deepcopy(models)
            fresh_opts = {v: torch.optim.AdamW(m.parameters(), lr=.01) for v, m in restored.items()}
            self.assertEqual(restore_group(path, restored, fresh_opts, protocol, 'cpu'), 1)
            advance(models, optimizers)
            advance(restored, fresh_opts)
            for v in models:
                for a, b in zip(models[v].parameters(), restored[v].parameters()):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
            with self.assertRaises(ValueError):
                restore_group(path, restored, fresh_opts, {'source': 'changed'}, 'cpu')

    def test_queue_has_three_matched_groups_and_no_disk_preparation(self):
        cfg = dict(checkpoint='/model', semantic_teacher='/teacher', seeds=[42, 43, 44],
                   deadline=1000, finish_reserve=100, steps=12000, batch_size=8, semantic_batch=8,
                   train_n=4096, val_n=128, test_n=447, human_n=138, save_every=100)
        jobs = plan(Path('/run'), cfg)
        self.assertEqual(len(jobs), 3)
        self.assertTrue(all(len(j['variants']) == 6 for j in jobs))
        self.assertNotIn('--cache', str(jobs))
        self.assertNotIn('medworld_spatial.prepare', str(jobs))
        self.assertEqual(ALLOWED_GPUS, ('1', '2', '3', '6', '7', '0'))
        with patch('medworld.gpu.subprocess.check_output') as query:
            for forbidden in ('4', '5'):
                with self.assertRaises(ValueError):
                    acquire_gpu(forbidden)
            query.assert_not_called()


if __name__ == '__main__':
    unittest.main()
