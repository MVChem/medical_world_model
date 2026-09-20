# Evaluation

`evaluate.py` evaluates classification, segmentation or VQA using a completed checkpoint. `compare_run.py` compares separately trained image-only and image-plus-slots models on matching complete cohorts.

`medworld.evaluate_run` reads the run's JSON `testing` block: enabled, selected tasks, human segmentation and validated result reuse. `medworld.run_experiment` trains both arms and automatically compares the configured tests. See the project README for commands.

All outputs stay under original run directories. Historical raw-Qwen/report/SR comparisons remain in their frozen runs; they are not part of the new three-task comparison.
