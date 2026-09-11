# Four-task Stage 1

The 2026-09-11 implementation is in `slot44_*.py`: clinical slots 1–4, spatial slots 5–8,
query-conditioned disease lists using all eight slots, and two spatial attention readouts for SR.
See the [new run notes](../../research_notes/0911_stage1_slot44_run.md) and
[live report](runs/slot44_20260911/REPORT.md). This preview uses explicitly named local
Chest ImaGenome derived QA; official MIMIC-CXR-VQA access is pending its project DUA.
The original implementation and completed run below remain available for historical reproduction.

Qwen3.5-0.8B + frozen V-JEPA 2.1 ViT-B; eight shared 1024-dimensional multimodal slots.
Current image and current report enter the state encoder. Classification and diagnosis read only slots.
Segmentation also reads the image; SR reads LR image and slots encoded from LR + report.

[Chinese plan](../../research_notes/0910_stage1_four_task_plan.md) · [Live overnight report](runs/overnight_20260910/REPORT.md)

The implementation is isolated from the existing Table-1 forecasting experiments. No treatment, IV linkage,
future pairing, Qwen change labels, or world-model prediction is used in this Stage-1 run.

1. `prepare.py`: candidate inventory over all 377,110 CXR images, official patient splits, deterministic
   independent-study sample, real JPEG decoding, aspect-preserving HR cache and task eligibility.
2. `cache.py features`: separately encode HR and synthetic LR with frozen V-JEPA. The two caches are distinct.
3. `cache.py segmentation`: original local CXAS architecture and checkpoint loaded strictly; save three
   FP16 organ-probability maps. Optional top-level CXAS visualization/DICOM packages are not imported.
4. `train.py`: four-task round robin, masked official-label BCE, report CE, anatomy distillation and SR MSE.
5. `evaluation.py`: held-out metrics, teacher-agreement segmentation, synthetic SR, state interventions,
   fixed-protocol frozen-slot linear probes, generated reports and clinical scoring subprocess.

`runner.py` is the durable coordinator for this run. It waits for complete caches, runs an end-to-end real-data
smoke using validation data, starts joint training and a matched-update frozen-encoder control on GPUs 1/5,
and updates REPORT.md every 30 seconds. GPU 6 prepares image features and GPU 7 prepares teacher masks.

```bash
tmux attach -t medworld_stage1_20260910
tail -f runs/overnight_20260910/joint.log
```

Training stops at the configured wall-clock deadline (2026-09-11 07:10 +08:00) or compute budget, saves
`checkpoint_final.pt`, and then evaluates. Saves also occur after the first complete four-task cycle,
every 30 minutes, and on validation improvement. SIGTERM/SIGINT save at the next optimizer boundary.
Resume with the same config and `train.py --run PATH --resume PATH/checkpoint_latest.pt`;
for the control preserve `--freeze-encoder --follow PRIMARY_PATH`.

Important scope: masks are teacher pseudo labels; SR scale=2 means H/2,W/2 → H,W (pixel count ×4),
HR long edge ≤512 with padding excluded from metrics. Class labels are report-derived and uncertain/blank
values are masked. No Finding is omitted because its released labels lack explicit negatives.
The test set is used only after training; a smoke run uses validation data instead.

Local checkpoints and data stay in ignored/non-versioned runtime directories; imported pilot model code
and this implementation are snapshotted under each run's source folder with fingerprints.
