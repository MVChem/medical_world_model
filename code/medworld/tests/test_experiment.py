import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from medworld.config import load_config
from medworld import run_experiment as experiment


class ExperimentTests(unittest.TestCase):
    def test_baseline_switches_are_independent_and_validated(self):
        self.assertEqual(load_config(overrides={'baselines': {'qwen': False}})['baselines'],
                         {'no_slots': True, 'qwen': False})
        for value in (None, {'unknown': True}, {'no_slots': 'false'}, {'qwen': 1}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                load_config(overrides={'baselines': value})

    def test_time_estimate_gives_slower_slots_more_time_but_identical_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, seconds in [('slots', 20), ('baseline', 10)]:
                directory = root / f'calibration_{name}'
                directory.mkdir()
                (directory / 'metrics.jsonl').write_text('\n'.join(json.dumps({
                    'task': ('classification', 'segmentation', 'vqa')[i % 3],
                    'wall_unix': 100 + (i + 1) * seconds}) for i in range(9)))
            plan = experiment.plan_updates(root, 3, 300)
            n = plan['target_steps_per_arm']
            self.assertEqual(n % 3, 0)
            self.assertGreater(n, 0)
            self.assertEqual(plan['estimated_training_seconds']['slots'],
                             2 * plan['estimated_training_seconds']['baseline'])
            cfg = load_config(overrides={'total_hours': 3})
            a, b = [experiment.arm_config(cfg, slots, n) for slots in (True, False)]
            self.assertEqual(a['steps'], b['steps'])
            self.assertEqual(a['total_hours'], b['total_hours'])
            self.assertEqual(a['total_hours'], 0)
            self.assertEqual(b['latent_weight'], 0)
            self.assertEqual(b['visual_consistency_weight'], 0)
            a.pop('latent_weight'); b.pop('latent_weight')
            a.pop('visual_consistency_weight'); b.pop('visual_consistency_weight')
            a.pop('slot_conditioning'); b.pop('slot_conditioning')
            self.assertEqual(a, b)
            self.assertEqual(cfg['total_hours'], 3)

    def run_fake(self, root, *, no_slots=True, qwen=True, hours=0, testing=True, mismatch=False, model_id="qwen08b"):
        run = root / 'experiment'
        from medworld_zero_shot_eval.models import models
        spec = next(s for s in models() if s['id'] == model_id)
        cfg = load_config(overrides={'steps': 6, 'total_hours': hours, 'qwen': spec['path'],
                          'baselines': {'no_slots': no_slots, 'qwen': qwen},
                          'testing': {'enabled': testing, 'tasks': ['classification', 'vqa'],
                                      'vqa_per_type': 100}})
        calls, settings = [], {}

        def launch(command, **kwargs):
            out = Path(command[command.index('--out') + 1])
            config = json.loads(Path(command[command.index('--config') + 1]).read_text())
            calls.append(out.name); settings[out.name] = config
            out.mkdir()
            count = config['steps'] + int(mismatch and out.name == 'baseline')
            (out / 'config.json').write_text(json.dumps(config))
            (out / 'status.json').write_text(json.dumps({'complete': True, 'step': count,
                                                        'started_unix': 100, 'heartbeat_unix': 200}))
            (out / 'final.pt').touch()
            (out / 'pipeline_status.json').write_text(json.dumps({'evaluation_complete': testing}))
            (out / 'evaluation_summary.json').write_text('{}')
            return Mock(wait=Mock(return_value=0), poll=Mock(return_value=0))

        def native(jobs, gpus, path, env):
            calls.append('qwen')
            args = jobs[0]['args']
            self.assertEqual(args[args.index('--model') + 1], model_id)
            self.assertNotIn('--limit', args)
            self.assertIn(str(run / 'slots/config.json'), args)

        with patch.object(experiment, 'freeze_experiment'), patch.object(experiment, 'registry'), \
             patch.object(experiment.subprocess, 'Popen', side_effect=launch), \
             patch.object(experiment, 'schedule', side_effect=native), \
             patch.object(experiment, 'plan_updates', return_value={'target_steps_per_arm': 12}), \
             patch('medworld.evaluation.compare_run.compare', return_value=[]), \
             patch('medworld.evaluation.compare_native.compare_native', return_value=[]):
            experiment.run_experiment(cfg, run, '2,3')
        return calls, settings, run

    def test_all_switch_combinations_preserve_slots_first_and_selected_tests(self):
        for no_slots in (False, True):
            for qwen in (False, True):
                with self.subTest(no_slots=no_slots, qwen=qwen), tempfile.TemporaryDirectory() as tmp:
                    calls, settings, run = self.run_fake(Path(tmp), no_slots=no_slots, qwen=qwen)
                    self.assertEqual(calls, ['slots'] + (['baseline'] if no_slots else []) + (['qwen'] if qwen else []))
                    if no_slots:
                        self.assertEqual(settings['slots']['steps'], settings['baseline']['steps'])
                        self.assertEqual(settings['slots']['task_batch_sizes'], settings['baseline']['task_batch_sizes'])
                        self.assertFalse(settings['baseline']['slot_conditioning'])
                        self.assertEqual(settings['baseline']['latent_weight'], 0)
                        self.assertEqual(settings['baseline']['visual_consistency_weight'], 0)
                    self.assertTrue(settings['slots']['slot_conditioning'])
                    self.assertEqual(json.loads((run / 'pipeline_status.json').read_text())['phase'], 'complete')

    def test_9b_schedules_matching_native_backbone(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls, settings, _ = self.run_fake(Path(tmp), model_id='qwen9b')
            self.assertEqual(calls, ['slots', 'baseline', 'qwen'])
            self.assertEqual(settings['slots']['qwen'], settings['baseline']['qwen'])
            self.assertEqual(experiment.native_spec(settings['slots'])['id'], 'qwen9b')

    def test_time_budget_calibrates_both_then_trains_equal_steps_from_scratch(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls, settings, _ = self.run_fake(Path(tmp), hours=3)
            self.assertEqual(calls, ['calibration_slots', 'calibration_baseline', 'slots', 'baseline', 'qwen'])
            for name in ('slots', 'baseline'):
                self.assertEqual(settings[name]['steps'], 12)
                self.assertEqual(settings[name]['total_hours'], 0)
                self.assertTrue(settings[name]['testing']['enabled'])
            for name in ('calibration_slots', 'calibration_baseline'):
                self.assertFalse(settings[name]['testing']['enabled'])
                self.assertEqual(settings[name]['steps'], 9)

    def test_single_timed_arm_uses_whole_budget_without_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls, settings, _ = self.run_fake(Path(tmp), hours=3, no_slots=False, qwen=False)
            self.assertEqual(calls, ['slots'])
            self.assertEqual(settings['slots']['total_hours'], 3)

    def test_disabled_testing_skips_native_but_still_trains_selected_arms(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls, _, _ = self.run_fake(Path(tmp), testing=False)
            self.assertEqual(calls, ['slots', 'baseline'])

    def test_mismatched_step_count_fails_before_native_or_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'update count'):
                self.run_fake(Path(tmp), mismatch=True)
            state = json.loads((Path(tmp) / 'experiment/pipeline_status.json').read_text())
            self.assertEqual(state['phase'], 'failed')


if __name__ == '__main__':
    unittest.main()
