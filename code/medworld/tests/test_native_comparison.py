import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from medworld.config import load_config
from medworld.datasets.protocol import _sha256
from medworld.evaluation.compare_native import compare_native
from medworld.evaluation.protocol import metric_protocol
from medworld.evaluation.selection import reference_manifest, select_vqa


class NativeComparisonTests(unittest.TestCase):
    def test_exact_records_hashes_selection_and_complete_status_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            slots, native = root / 'slots', root / 'qwen'
            evaluation = slots / 'evaluation/vqa'
            evaluation.mkdir(parents=True); native.mkdir()
            (slots / 'data_protocol.json').write_text('{}')
            (slots / 'final.pt').write_bytes(b'checkpoint')
            cfg = load_config()
            (slots / 'config.json').write_text(json.dumps(cfg))
            fingerprint = hashlib.sha256(b'{}').hexdigest()
            record = {'id': 'vqa:test:1', 'patient': '7', 'question': 'Question?',
                      'answer': ['yes'], 'semantic_type': 'verify', 'prediction': '["yes"]'}
            for folder in (evaluation, native):
                (folder / 'vqa.jsonl').write_text(json.dumps(record) + '\n')
            _, selection = select_vqa([record])
            summary = {'metric_protocol': metric_protocol('table2'),
                       'references': {'vqa': reference_manifest('vqa', [record])},
                       'limit': None, 'split': 'test', 'data_fingerprint': fingerprint,
                       'vqa_selection': selection, 'tasks': {'vqa': {'n': 1, 'exact_match': 1, 'micro_f1': 1}},
                       'predictions_sha256': {'vqa': _sha256(evaluation / 'vqa.jsonl')}}
            trained = {**summary, 'checkpoint_sha256': _sha256(slots / 'final.pt')}
            raw = {**summary, 'model_id': 'qwen08b', 'model': cfg['qwen'], 'partial': False}
            (evaluation / 'summary.json').write_text(json.dumps(trained))
            (native / 'summary.json').write_text(json.dumps(raw))
            (native / 'status.json').write_text('{"status":"complete"}')
            self.assertEqual(len(compare_native(slots, native, root, ['vqa'])), 2)
            from medworld_zero_shot_eval.models import models
            nine = next(s for s in models() if s['id'] == 'qwen9b')
            (slots / 'config.json').write_text(json.dumps({**cfg, 'qwen': nine['path']}))
            with self.assertRaisesRegex(ValueError, 'Qwen3.5-9B'):
                compare_native(slots, native, root, ['vqa'])
            raw.update(model_id='qwen9b', model=nine['path'])
            (native / 'summary.json').write_text(json.dumps(raw))
            self.assertEqual(len(compare_native(slots, native, root, ['vqa'])), 2)
            (native / 'vqa.jsonl').write_text(json.dumps({**record, 'id': 'another'}) + '\n')
            with self.assertRaisesRegex(ValueError, 'hash'):
                compare_native(slots, native, root, ['vqa'])
            raw['predictions_sha256'] = {'vqa': _sha256(native / 'vqa.jsonl')}
            (native / 'summary.json').write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, 'IDs or references'):
                compare_native(slots, native, root, ['vqa'])
            (native / 'vqa.jsonl').write_text(json.dumps(record) + '\n')
            raw['predictions_sha256'] = {'vqa': _sha256(native / 'vqa.jsonl')}
            raw['vqa_selection'] = {**selection, 'seed': 7}
            (native / 'summary.json').write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, 'sampling protocol'):
                compare_native(slots, native, root, ['vqa'])
            (native / 'status.json').write_text('{"status":"failed"}')
            with self.assertRaisesRegex(ValueError, 'complete'):
                compare_native(slots, native, root, ['vqa'])


if __name__ == '__main__':
    unittest.main()
