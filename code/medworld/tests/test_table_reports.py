import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from medworld.config import load_config
from medworld.datasets.protocol import _sha256
from medworld.datasets.vqa import VOCABULARY
from medworld.evaluation.future_common import scoring_protocol
from medworld.evaluation.future_metrics import SCHEMA, reference_fingerprint, score_future_task
from medworld.evaluation.protocol import FUTURE_TASKS, TABLE1_METRICS, metric_protocol
from medworld.evaluation.selection import reference_manifest, select_vqa
from medworld.evaluation.table_reports import write_tables
from medworld.runtime import source_fingerprint
from medworld_zero_shot_eval.future_evaluate import native_readout_protocol


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class TableReportTests(unittest.TestCase):
    def fixture(self, root):
        weights, jepa, assets = root / 'weights', root / 'jepa.pt', root / 'radgraph'
        weights.mkdir()
        (weights / 'model.safetensors').write_bytes(b'fake qwen weights')
        jepa.write_bytes(b'fake jepa weights')
        cfg = load_config(overrides={'future_enabled': True, 'qwen': str(weights), 'jepa': str(jepa),
                                     'radgraph_assets': str(assets), 'total_hours': 0})
        write(assets / 'manifest.json', {'provenance': {'package_version': '0.1.18', 'model_type': 'radgraph-xl',
              'reward_level': 'partial', 'report_scope': 'full_report',
              'model_files_sha256': {'weights.th': 'a' * 64, 'config.json': 'b' * 64},
              'scorer_files_sha256': {name: 'c' * 64 for name in ('radgraph.py', 'rewards.py', 'utils.py')}}})
        data_protocol = {'future': {'cohort': 'test fixture'}}
        fingerprint = hashlib.sha256(json.dumps(data_protocol, sort_keys=True).encode()).hexdigest()
        future_fingerprint = hashlib.sha256(json.dumps(data_protocol['future'], sort_keys=True).encode()).hexdigest()
        checkpoints = {}
        for arm in ('baseline', 'slots'):
            settings = load_config(overrides={**cfg, 'slot_conditioning': arm == 'slots'})
            (root / arm).mkdir()
            (root / arm / 'final.pt').write_bytes(arm.encode())
            checkpoints[arm] = {'config': settings, 'data_fingerprint': fingerprint,
                                'weights_fingerprint': source_fingerprint(settings),
                                'metadata': {'task_initialization_sha256': 'a' * 64},
                                'progress': {'complete': True, 'step': 8, 'world_size': 2,
                                             'task_samples': {task: 2 for task in ('classification', 'vqa', 'segmentation', *FUTURE_TASKS)}}}
        native_base = {'model_id': 'qwen9b', 'model': str(weights), 'partial': False,
                       'weights_sha256': {'model.safetensors': _sha256(weights / 'model.safetensors')}}
        native = {table: {**native_base, 'metric_protocol': metric_protocol(table), 'split': 'test', 'limit': None,
                          'data_fingerprint': fingerprint, 'future_data_fingerprint': future_fingerprint,
                          'tasks': {}, 'references': {}, 'predictions_sha256': {}} for table in ('table1', 'table2')}
        native['table1']['readout_protocol'] = native_readout_protocol(cfg)
        for folder in ('baseline', 'slots', 'qwen', 'qwen_future'):
            write(root / folder / 'data_protocol.json', data_protocol)
        for folder in ('qwen', 'qwen_future'):
            write(root / folder / 'status.json', {'status': 'complete', 'partial': False})
        for task, key in TABLE1_METRICS.items():
            targets = ({'future_vqa': [VOCABULARY[0]] * 3, 'progression': ['improved', 'stable', 'worsened'],
                        'future_report': ['Report text'] * 3, 'mortality_30d': [0, 1, 0], 'remaining_los': [1., 2., 3.]})[task]
            references = [{'id': f'{task}:{index}', 'patient': str(index), 'target': target,
                           **({'finding': 'edema'} if task == 'progression' else {})}
                          for index, target in enumerate(targets)]
            protocol = scoring_protocol(task, references, cfg)
            if task == 'future_report':
                score = {'schema': SCHEMA, 'task': task, 'unit': 'fraction',
                         'n': 3, 'complete': True, key: .6, 'reference_sha256': reference_fingerprint(references),
                         'protocol_sha256': hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(',', ':')).encode()).hexdigest()}
            else:
                score = score_future_task(task, references, {r['id']: r['target'] for r in references}, protocol=protocol)
            for arm in ('baseline', 'slots', 'qwen'):
                folder = root / ('qwen_future' if arm == 'qwen' else f'{arm}/evaluation/{task}')
                folder.mkdir(parents=True, exist_ok=True)
                records = [{**row, 'prediction': row['target']} for row in references]
                (folder / f'{task}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in records))
                summary = native['table1'] if arm == 'qwen' else {
                    'metric_protocol': metric_protocol('table1'), 'split': 'test', 'limit': None,
                    'data_fingerprint': fingerprint, 'future_data_fingerprint': future_fingerprint,
                    'checkpoint_sha256': _sha256(root / arm / 'final.pt'),
                    'tasks': {}, 'references': {}, 'predictions_sha256': {}}
                summary['tasks'][task] = score
                summary['references'][task] = {'n': len(records), 'references_sha256': reference_fingerprint(references)}
                summary['predictions_sha256'][task] = _sha256(folder / f'{task}.jsonl')
                write(folder / f'{task}_protocol.json', protocol)
                write(folder / 'summary.json', summary)
        for task in ('classification', 'vqa', 'segmentation', 'segmentation_human'):
            actual = 'segmentation' if task == 'segmentation_human' else task
            groups = None
            if actual == 'classification':
                records = [{'id': 'c1', 'patient': 'p1', 'labels': [1, 0], 'probabilities': [.9, .1]}]
                metrics = {'n': 1, 'macro_auroc': .7, 'macro_ap': .6}
            elif actual == 'vqa':
                records = [{'id': 'v1', 'patient': 'p1', 'question': 'Question?', 'answer': ['yes'],
                            'semantic_type': 'verify', 'prediction': '["yes"]'}]
                metrics = {'n': 1, 'exact_match': 1., 'micro_f1': 1.}
            else:
                datasets = ['montgomery'] if task == 'segmentation_human' else metric_protocol('table2')['segmentation_main_datasets']
                records = [{'id': dataset, 'patient': 'p1', 'dataset': dataset, 'volume_id': dataset,
                            'active_channels': [0], 'target_names': ['organ'], 'target_sha256': 'b' * 64} for dataset in datasets]
                groups = {dataset: {'n_images': 1, 'n_volumes': 1, 'n_patients': 1, 'active_channels': [0],
                                    'target_names': ['organ'], 'mean_dice': .6, 'mean_iou': .5} for dataset in datasets}
                metrics = {'n': len(records), 'mean_dice': .6, 'mean_iou': .5, 'by_dataset': groups}
            for arm in ('baseline', 'slots', 'qwen'):
                if arm == 'qwen' and actual == 'segmentation':
                    continue
                folder = root / ('qwen' if arm == 'qwen' else f'{arm}/evaluation/{task}')
                folder.mkdir(parents=True, exist_ok=True)
                (folder / f'{actual}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in records))
                summary = native['table2'] if arm == 'qwen' else {
                    'metric_protocol': metric_protocol('table2'), 'split': 'human_test' if task == 'segmentation_human' else 'test', 'limit': None,
                    'data_fingerprint': fingerprint, 'checkpoint_sha256': _sha256(root / arm / 'final.pt'),
                    'tasks': {}, 'references': {}, 'predictions_sha256': {}}
                summary['tasks'][actual] = metrics
                summary['references'][actual] = reference_manifest(actual, records)
                summary['predictions_sha256'][actual] = _sha256(folder / f'{actual}.jsonl')
                if actual == 'vqa':
                    summary['vqa_selection'] = select_vqa(records)[1]
                write(folder / 'summary.json', summary)
        return cfg, checkpoints

    def export(self, root, cfg, checkpoints):
        with patch('medworld.evaluation.table_reports.read_checkpoint', side_effect=lambda path: checkpoints[path.parent.name]), \
             patch('medworld.run_experiment.native_spec', return_value={'id': 'qwen9b'}):
            return write_tables(root, cfg)

    def test_exports_both_complete_tables_and_external_segmentation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg, checkpoints = self.fixture(root)
            result = self.export(root, cfg, checkpoints)
            self.assertIsNone(result['table2']['qwen']['mean_dice'])
            self.assertEqual(set(result['segmentation_by_dataset']['baseline']),
                             {'mimic_cxr_human', 'ucsf_alptdg', 'mu_glioma_post', 'montgomery'})
            self.assertEqual(result['table1']['slots']['mae_days'], 0.)
            self.assertEqual((root / 'COMPARISON.md').read_text().count('Qwen3.5-9B zero-shot'), 2)
            self.assertIn('N/A', (root / 'TABLE2.md').read_text())
            self.assertTrue((root / 'TABLE1.md').is_file())
            self.assertTrue((root / 'tables.json').is_file())

    def test_rejects_incomplete_or_changed_artifacts_before_export(self):
        def mutate(root, location, change):
            path = root / location
            value = json.loads(path.read_text())
            change(value)
            path.write_text(json.dumps(value))
        cases = [
            lambda root, saved: saved['slots']['progress']['task_samples'].update(future_report=3),
            lambda root, saved: mutate(root, 'qwen_future/summary.json', lambda s: s['tasks']['future_vqa'].update(accuracy=None)),
            lambda root, saved: mutate(root, 'qwen/summary.json', lambda s: s['weights_sha256'].update(**{'model.safetensors': 'bad'})),
            lambda root, saved: mutate(root, 'slots/evaluation/segmentation/summary.json', lambda s: s['tasks']['segmentation']['by_dataset'].pop('ucsf_alptdg')),
            lambda root, saved: mutate(root, 'slots/evaluation/segmentation/summary.json', lambda s: s['tasks']['segmentation'].update(mean_dice=.9)),
            lambda root, saved: mutate(root, 'baseline/evaluation/progression/summary.json', lambda s: s.pop('metric_protocol')),
            lambda root, saved: (root / 'qwen_future/future_vqa.jsonl').write_text('{}\n'),
        ]
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                cfg, checkpoints = self.fixture(root)
                case(root, checkpoints)
                with self.assertRaises(ValueError):
                    self.export(root, cfg, checkpoints)
                self.assertFalse((root / 'tables.json').exists())

    def test_rejects_self_consistent_but_different_future_references(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg, checkpoints = self.fixture(root)
            folder, task = root / 'qwen_future', 'future_vqa'
            records = [json.loads(line) for line in (folder / f'{task}.jsonl').read_text().splitlines()]
            records[0]['patient'] = 'a different patient'
            references = [{key: value for key, value in row.items() if key != 'prediction'} for row in records]
            protocol = scoring_protocol(task, references, cfg)
            score = score_future_task(task, references, {row['id']: row['prediction'] for row in records}, protocol=protocol)
            (folder / f'{task}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in records))
            summary = json.loads((folder / 'summary.json').read_text())
            summary['tasks'][task] = score
            summary['references'][task] = {'n': len(records), 'references_sha256': reference_fingerprint(references)}
            summary['predictions_sha256'][task] = _sha256(folder / f'{task}.jsonl')
            write(folder / f'{task}_protocol.json', protocol)
            write(folder / 'summary.json', summary)
            with self.assertRaisesRegex(ValueError, 'identical ordered references'):
                self.export(root, cfg, checkpoints)
            self.assertFalse((root / 'tables.json').exists())


if __name__ == '__main__':
    unittest.main()
