# Evaluation

`evaluate.py` evaluates classification, segmentation or VQA using a completed checkpoint. `compare_run.py` compares separately trained image-only and image-plus-slots models on matching complete cohorts.

`medworld.evaluate_run` reads the run's JSON `testing` block: enabled, selected tasks, human segmentation and validated result reuse. `medworld.run_experiment` trains both arms and automatically compares the configured tests. See the project README for commands.

The experiment runs the slots model first, followed by optional no-slots training and native Qwen3.5-0.8B tests, controlled by `baselines.no_slots` and `baselines.qwen`. Trainable arms must complete equal optimizer updates. A requested time budget is converted to a common step count after short timing probes; evaluation time is additional. `compare_native.py` verifies final-checkpoint provenance, prediction hashes, test IDs/references and VQA selection before combining native and trained scores.

All outputs stay under original run directories. Historical raw-Qwen/report/SR comparisons remain in their frozen runs; they are not part of the new three-task comparison.
