import json
from pathlib import Path
import tempfile
import unittest

from medworld.config import load_config
from medworld.evaluation.protocol import FUTURE_TASKS, training_tasks
from medworld.run_experiment import plan_updates, write_report


class FuturePipelineContractTests(unittest.TestCase):
    def test_24_step_calibration_discards_one_complete_eight_task_cycle(self):
        cfg = load_config(overrides={'future_enabled': True})
        tasks = training_tasks(cfg)
        self.assertEqual(len(tasks), 8)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for arm, multiplier in [('slots', 2), ('baseline', 1)]:
                path = root / f'calibration_{arm}/metrics.jsonl'
                path.parent.mkdir()
                wall, records = 0., []
                for index in range(24):
                    wall += 1000 if index < 8 else multiplier * (index % 8 + 1)
                    records.append({'task': tasks[index % 8], 'wall_unix': wall})
                path.write_text(''.join(json.dumps(row) + '\n' for row in records))
            plan = plan_updates(root, 24, 600, tasks)
            self.assertEqual(plan['seconds_per_step'], {'slots': 9., 'baseline': 4.5})
            self.assertEqual(plan['target_steps_per_arm'] % 8, 0)
            self.assertFalse(plan['evaluation_time_included'])
            path = root / 'calibration_slots/metrics.jsonl'
            path.write_text(path.read_text().replace('"future_report"', '"classification"'))
            with self.assertRaisesRegex(ValueError, 'all training tasks'):
                plan_updates(root, 24, 600, tasks)

    def test_intermediate_current_report_does_not_treat_future_tasks_as_current_metrics(self):
        cfg = load_config(overrides={'future_enabled': True})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'slots').mkdir()
            summaries = {task: {'tasks': {task: {'n': 3}}} for task in FUTURE_TASKS}
            summaries['classification'] = {'tasks': {'classification': {'n': 2, 'macro_auroc': .7, 'macro_ap': .6}}}
            (root / 'slots/evaluation_summary.json').write_text(json.dumps(summaries))
            budget = {'steps': 8, 'target_steps': 8, 'elapsed_seconds': 10.}
            write_report(root, cfg, {'slots': budget, 'no_slots': budget}, [], [], {})
            comparison = json.loads((root / 'comparison.json').read_text())
            self.assertEqual({row['task'] for row in comparison['rows']}, {'classification'})


if __name__ == '__main__':
    unittest.main()
