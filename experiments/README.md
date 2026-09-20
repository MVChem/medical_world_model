# Experiments

## Demos

| Demo | Description | Open | Source and setup |
|---|---|---|---|
| MIMIC Atlas demo · 20260918 | Minimal runnable React + FastAPI snapshot; full data and indexes linked from the original project. | [Overview](http://127.0.0.1:8768/#overview) · [Featured cases](http://127.0.0.1:8768/#featured) | [Demo and startup guide](mimic_atlas_demo_20260918/README.md) |
| MIMIC Atlas | React + FastAPI browser linking MIMIC-CXR and MIMIC-IV: patient timelines, paired images, reports, and clinical records. | [Overview](http://127.0.0.1:8767/#overview) · [Featured cases](http://127.0.0.1:8767/#featured) | [Project and startup guide](../code/mimic_atlas/README.md) |

Start the snapshot with `./experiments/mimic_atlas_demo_20260918/start.sh` from the repository root (port 8768). The original service uses port 8767 and is managed by `mimic-atlas.service`. For remote viewing, forward the selected port with SSH; see the linked startup guides.

## Experiment history

Ongoing experiments: [registry.json](registry.json).
Details remain in the linked run directories. When a run ends, move its entry here with its date, outcome, and run link.

| Date | ID | Outcome | Run |
|---|---|---|---|
| 2026-09-18 | `sr_area64_only_seed42_20260918` | Completed: six variants, 2,000 updates each; evaluation complete. | [Run and results](../code/medworld_spatial/runs/sr_area64_only_seed42_20260918/) |
| 2026-09-18 | `sr_area16_only_seed42_20260918` | Completed: six variants, 2,000 updates each; evaluation complete. | [Run and results](../code/medworld_spatial/runs/sr_area16_only_seed42_20260918/) |
| 2026-09-18 | `featup_online_seed42_20260918` | Completed at 17:03 CST: all six variants reached 4,000 updates; validation, test, and human-test evaluations complete. | [Run and results](../code/medworld_spatial/runs/featup_online_seed42_20260918/) |
| 2026-09-18 | `patient_index_20260918` | Completed and validated: global MIMIC-IV/CXR index for 368,138 patients; 11.47 GiB cache. | [Run and results](../code/mimic_atlas/runs/patient_index_20260918/) |
| 2026-09-18 | `unpack_mimic_iv_20260918` | Completed: 31 CSVs (90.52 GiB); verified and removed gzip originals (9.92 GiB). Service checked. | [Run](../code/mimic_atlas/runs/unpack_mimic_iv_20260918/) |
| 2026-09-18 | `csv_offset_index_20260918` | Completed: 354 MiB offsets for 368,138 patients; React/FastAPI deployed, raw-data/browser checks passed, 11.47 GiB legacy copies removed. | [Run and results](../code/mimic_atlas/runs/patient_index_20260918_offsets/) |
| 2026-09-18 | `medworld_featup_integration_smoke_20260918` | Failed before first update: retired pixel arrays; added source-image fallback before retry. | Deleted at user request on 2026-09-18; former run: `code/medworld/runs/featup_integration_smoke_20260918/` |
| 2026-09-18 | `medworld_featup_integration_smoke_v2_20260918` | Interrupted after Stage 1 and 3 temporal updates to enforce source-only pixel access. | Deleted at user request on 2026-09-18; former run: `code/medworld/runs/featup_integration_smoke_v2_20260918/` |
| 2026-09-18 | `medworld_featup_source_smoke_20260918` | Completed: source-only four-task Stage 1, 16 Stage 2 updates/all replays; gradient and reload audits passed; 21 tests passed. | Deleted at user request on 2026-09-18; former run: `code/medworld/runs/featup_source_smoke_20260918/` |
| 2026-09-18 | `capacity_featup_20260918` | Completed: batch probes; native-Qwen test pixels/references verified, baseline reports re-scored with current CheXbert; 22 tests passed. | Deleted at user request on 2026-09-19; former run: `code/medworld/runs/capacity_featup_20260918/` |
| 2026-09-18 | `featup_4gpu_preflight_20260918` | Interrupted: Stage 1 passed; Stage 2 report replay stalled at near-full memory on rank 1. Reduced temporal/report batches for retry. | Deleted at user request on 2026-09-19; former run: `code/medworld/runs/featup_4gpu_preflight_20260918/` |
| 2026-09-18 | `featup_4gpu_preflight_v2_20260918` | Completed: all four Stage 1 tasks, eight Stage 2 updates/two replay cycles, validation and checkpoints; replica spread 0. | Deleted at user request on 2026-09-19; former run: `code/medworld/runs/featup_4gpu_preflight_v2_20260918/` |
| 2026-09-19 | `qwen35_08b_featup_4gpu_8h_20260918` | Completed: eight-hour two-stage training and full downstream tests; artifacts subsequently deleted at user request. | Deleted at user request on 2026-09-19; former run: `code/medworld/runs/qwen35_08b_featup_4gpu_8h_20260918/` |
| 2026-09-19 | `capacity_20260919` | Passed real-data capacity probe: five tasks, temporal report replay, batch 64; peak allocated 14.7 GiB. | [Run](../code/medworld/runs/capacity_20260919/) |
| 2026-09-19 | `preflight_4gpu_20260919` | Interrupted at user request before temporal training; eight-hour run cancelled. | [Run](../code/medworld/runs/preflight_4gpu_20260919/) |
| 2026-09-19 | `joint_preflight_20260919` | Passed four-GPU joint preflight, checkpoint and all-task validation. | [Run](../code/medworld/runs/joint_preflight_20260919/) |
| 2026-09-19 | `qwen35_08b_joint_8h_20260919` | Completed: full held-out tests and fresh native Qwen3.5-0.8B comparison. | [Run](../code/medworld/runs/qwen35_08b_joint_8h_20260919/) |
| 2026-09-19 | `qwen35_08b_joint_8h_20260919_native_baseline` | Completed: fresh native Qwen3.5-0.8B tests and matched comparison; see parent COMPARISON.md. | [Run](../code/medworld/runs/qwen35_08b_joint_8h_20260919/qwen35_08b_joint_8h_20260919_native_baseline/) |
| 2026-09-20 | `three_tasks_smoke_20260920` | Passed: 36 tests; three-task real-data baseline/slots gradients and reload, two-sample inference, two-GPU training (replica spread 0), JSON testing disable verified. | [Run](../code/medworld/runs/three_tasks_smoke_20260920/) |
| 2026-09-20 | `native_four_models_20260920` | Interrupted at user request; replaced by serial test-only evaluation. | Original run `code/medworld_zero_shot_eval/runs/native_four_models_20260920/` deleted at user request. |
| 2026-09-20 | `native_serial_test_20260920` | Interrupted: replaced by balanced 300-question VQA protocol. | Superseded run deleted per requested run cleanup. |
| 2026-09-20 | `native_serial_vqa300_20260920` | Failed or partial; see per-model status and logs. | [Run](../code/medworld_zero_shot_eval/runs/native_serial_vqa300_20260920/) |
