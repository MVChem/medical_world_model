import unittest
import torch

class PixelFallbackTests(unittest.TestCase):
    def test_source_only_ignores_pixel_caches(self):
        import tempfile
        from pathlib import Path
        import numpy as np
        from PIL import Image
        from medworld.datasets.current import MultiTaskData
        from medworld.datasets.pixels import source_canvas, human_target
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / 'source.png'
            Image.fromarray(np.random.default_rng(7).integers(0, 256, (80, 60), dtype=np.uint8)).save(image)
            mask = root / 'mask.png'
            Image.fromarray(np.full((80, 60), 255, dtype=np.uint8)).save(mask)
            source = {'image': str(image), 'box': [0, 64, 512, 384], 'masks': [str(mask), str(mask)]}
            data = MultiTaskData.__new__(MultiTaskData)
            data.old, data.dense = root / 'old', root / 'dense'
            data.old.mkdir(); data.dense.mkdir()
            data._old_count = data._dense_count = 1
            data._old_sources = data._dense_sources = [source]
            data._arrays = {'pseudo': np.ones((1, 3, 256, 256), dtype=np.float16)}
            hr = (source_canvas(source)[0] * 255).round().byte().numpy()
            human = human_target(source).byte().numpy()
            paths = [(data.old / 'images.npy', hr), (data.dense / 'images.npy', hr),
                     (data.dense / 'human_masks.npy', human)]
            for path, array in paths:
                np.save(path, array[None])
            row = {'id': 'case', 'subject_id': 'patient', 'image_index': 0, 'index': 0,
                   'old_index': 0, 'human_index': 0, 'box': source['box'], 'kind': 'montgomery',
                   'report_target': 'reference', 'labels': [0] * 13}
            from unittest.mock import patch
            with patch('numpy.load', side_effect=AssertionError('Pixel cache accessed')):
                cached = {task: data._example(task, 'test', row) for task in
                          ('classification', 'segmentation')}
            torch.testing.assert_close(cached['segmentation']['targets'], torch.from_numpy(human).float())
            data._arrays = {'pseudo': data._arrays['pseudo']}
            for path, _ in paths:
                path.unlink()
            for task, expected in cached.items():
                actual = data._example(task, 'test', row)
                for key in ('pixels', 'targets', 'mask'):
                    if key in expected:
                        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
                np.testing.assert_array_equal(actual['image'], expected['image'])
            self.assertFalse(any(data.dense.glob('*.npy')))


if __name__ == "__main__":
    unittest.main()
