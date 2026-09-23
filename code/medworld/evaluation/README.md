# Evaluation

`evaluate.py` evaluates classification, segmentation or VQA using a completed
checkpoint. `compare_run.py` compares separately trained raw-input and
raw-input-plus-slots models on matching complete cohorts. Both use identical
downstream initialization, recorded by `task_initialization_sha256`. Only the
slots model has a Qwen/JEPA state encoder and representation-learning objectives.

`medworld.evaluate_run` reads the run's JSON `testing` block: enabled, selected tasks, human segmentation and validated result reuse. `medworld.run_experiment` trains both arms and automatically compares the configured tests. See the project README for commands.

The experiment runs the slots model first, followed by optional raw-input baseline
training and native Qwen tests, controlled by `baselines.no_slots` and
`baselines.qwen`. Native Qwen uses the configured slot-backbone size; the raw-input
baseline's parameter count does not depend on that size. Trainable arms must
complete equal optimizer updates. A requested time budget is converted to a
common step count after short timing probes; evaluation time is additional.
Recalibrate the new architecture rather than reusing historical throughput.
`compare_native.py` verifies final-checkpoint provenance, prediction hashes, test
IDs/references and VQA selection before combining native and trained scores.

Completed new experiments report classification AUROC/AP, VQA label-set exact
match/micro-F1, and segmentation mean IoU/Dice for both trained arms. Use only
human-reviewed segmentation: the 200-image MIMIC heart/lung cohort,
UCSF-ALPTDG and MU-Glioma-Post MRI, plus the external Montgomery human lung test.
Report each segmentation dataset separately and aggregate MRI slices by volume.
Native Qwen has no segmentation head: its segmentation metrics are N/A.

All outputs stay under original run directories. Previous 0.8B/9B no-slot-input
ablation scores do not measure the current raw-input baseline. Evaluation
supports only current checkpoints and the manual v2 data protocol; formal scores
require new training.
