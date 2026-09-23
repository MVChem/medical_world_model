# Experiments

## Evaluation archives

- [Slots vs no-slots evaluation · 20260921](slots_segmentation_evaluation_20260921/README.md): Codex Chat ID, segmentation Dice/IoU, classification and VQA results, and source links.

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
| 2026-09-20 | `throughput_probe_20260920` | Completed: batch scaling 1x/2x/4x, three tasks with temporal supervision. | Deleted at user request on 2026-09-20; former run: `code/medworld/runs/throughput_probe_20260920/` |
| 2026-09-20 | `native_retest_vqa300_20260920` | Completed: all four native models evaluated. | [Run](../code/medworld_zero_shot_eval/runs/native_retest_vqa300_20260920/) |
| 2026-09-20 | `qwen35_08b_vssc_2gpu_3h_20260920` | Completed configured tests: classification, vqa. | Deleted at user request on 2026-09-20; former run: `code/medworld/runs/qwen35_08b_vssc_2gpu_3h_20260920/` |
| 2026-09-20 | `paired_vssc_2gpu_16h_20260920` | Failed; see pipeline_status.json and original run logs. | Deleted at user request on 2026-09-20; former run: `code/medworld/runs/paired_vssc_2gpu_16h_20260920/` |
| 2026-09-20 | `throughput_tuning_20260920` | Completed: 49 tests, preprocessing/attention checks and two-GPU throughput probes; fresh 12-hour run launched. | [Run](../code/medworld/runs/throughput_tuning_20260920/) |
| 2026-09-20 | `paired_vssc_2gpu_12h_fast_20260920_slots` | Completed configured tests: classification, vqa. | [Run](../code/medworld/runs/paired_vssc_2gpu_12h_fast_20260920/slots/) |
| 2026-09-21 | `paired_vssc_2gpu_12h_fast_20260920_baseline` | Completed configured tests: classification, vqa. | [Run](../code/medworld/runs/paired_vssc_2gpu_12h_fast_20260920/baseline/) |
| 2026-09-21 | `paired_vssc_2gpu_12h_fast_20260920` | Completed: slots and no-slots training; selected tests and native Qwen comparison. | [Run](../code/medworld/runs/paired_vssc_2gpu_12h_fast_20260920/) |
| 2026-09-21 | `segmentation_backfill_20260921` | Completed full pseudo-mask (447) and human-mask (138) segmentation tests for both trained arms; Dice and IoU merged into parent comparison. | [Run](../code/medworld/runs/paired_vssc_2gpu_12h_fast_20260920/segmentation_backfill_20260921/) |
| 2026-09-21 | `qwen35_9b_preflight_20260921` | Passed: 51 tests, real-data capacity, two-GPU three-task training, reload and native 9B inference; both mask evaluations checked. | [Run](../code/medworld/runs/qwen35_9b_preflight_20260921/) |
| 2026-09-22 | `paired_qwen35_9b_2gpu_48h_20260921_slots` | Completed configured tests: classification, segmentation, vqa, segmentation_human. | [Run](../code/medworld/runs/paired_qwen35_9b_2gpu_48h_20260921/slots/) |
| 2026-09-22 | `fig5_v2_reproduction_20260922` | Failed: see original build log; no training. | [Run](../code/fig5_v2/runs/reproduction_20260922/) |
| 2026-09-22 | `fig5_v2_reproduction_v2_20260922` | Completed: two real MRI cases, verified slot attention, exploratory feature change; inference only. | [Run](../code/fig5_v2/runs/reproduction_v2_20260922/) |
| 2026-09-22 | `fig5_v2_reproduction_9b_20260922` | Completed: two real MRI cases, verified slot attention, exploratory feature change; inference only. | [Run](../code/fig5_v2/runs/reproduction_9b_20260922/) |
| 2026-09-22 | `fig6_real_20260922` | Completed: real image-slot retrieval for 457 examinations; PNG/SVG/PDF and local HTML. | [Run](../code/fig6_v1/runs/fig6_real_20260922/) |
| 2026-09-22 | `fig6_real_v2_20260922` | Completed: real image-slot retrieval for 457 examinations; PNG/SVG/PDF and local HTML. | [Run](../code/fig6_v1/runs/fig6_real_v2_20260922/) |
| 2026-09-22 | `latent_audit_20260922` | Failed: see latent audit run log and status. | [Run](../code/fig6_v1/runs/latent_audit_20260922/) |
| 2026-09-22 | `latent_audit_08b_20260922` | Completed: 457 examinations, 11 representations and 12 unsupervised projections; metrics and grid. | [Run](../code/fig6_v1/runs/latent_audit_08b_20260922/) |
| 2026-09-22 | `fig5_v2_attention_calibration_20260922` | Completed: frozen-encoder MRI attention calibration, patient-held-out evaluation and figure. | [Run](../code/fig5_v2/runs/attention_calibration_20260922/) |
| 2026-09-22 | `latent_audit_9b_20260922` | Completed: 457 examinations, 11 representations and 12 unsupervised projections; metrics and grid. | [Run](../code/fig6_v1/runs/latent_audit_9b_20260922/) |
| 2026-09-22 | `latent_audit_08b_report_20260922` | Completed: 457 examinations, 11 representations and 12 unsupervised projections; metrics and grid. | [Run](../code/fig6_v1/runs/latent_audit_08b_report_20260922/) |
| 2026-09-22 | `forecast_probe_20260922` | Completed: paired frozen-model temporal probes, held-out forecasts and real case figure; inherited task hashes verified. | [Run](../code/fig4_v1/runs/forecast_probe_20260922/) |
| 2026-09-22 | `fig5_v2_attention_export_20260922` | Completed: trained MRI attention export and original feature change; inference only. | [Run](../code/fig5_v2/runs/attention_export_20260922/) |
| 2026-09-22 | `latent_audit_9b_report_20260922` | Completed: 457 examinations, 11 representations and 12 unsupervised projections; metrics and grid. | [Run](../code/fig6_v1/runs/latent_audit_9b_report_20260922/) |
| 2026-09-22 | `forecast_case_replay_20260922` | Completed: saved temporal probes reloaded; held-out case predicted and rendered without retraining. | [Run](../code/fig4_v1/runs/forecast_case_replay_20260922/) |
| 2026-09-22 | `forecast_9b_2048_20260922` | Completed: frozen 9B slots temporal probe, all held-out forward pairs and real case; original task hashes verified. | [Run](../code/fig4_v1/runs/forecast_9b_2048_20260922/) |
| 2026-09-22 | `forecast_9b_replay_20260922` | Completed: saved 9B probe reloaded; all 24 fixed-case probabilities reproduced without training. | [Run](../code/fig4_v1/runs/forecast_9b_replay_20260922/) |
| 2026-09-22 | `clinical_profile_9b_20260922` | Completed: actual 13-finding logits/probabilities, two fixed UMAPs and decoder-norm diagnostics; inference only. | [Run](../code/fig6_v1/runs/clinical_profile_9b_20260922/) |
| 2026-09-22 | `fig5_v2_mri_forecast_20260922` | Completed: MRI probe; no clear held-out gain over persistence, weak change localization. | [Run](../code/fig5_v2/runs/mri_forecast_20260922/) |
| 2026-09-22 | `fig5_v2_mri_forecast_export_20260922` | Completed: MRI transition figure and probability components; inference only. | [Run](../code/fig5_v2/runs/mri_forecast_export_20260922/) |
| 2026-09-22 | `multimodal_tokens_9b_20260922` | Completed: original image/current-report language token diagnostics; metrics and comparison figure. | [Run](../code/fig6_v1/runs/multimodal_tokens_9b_20260922/) |
| 2026-09-22 | `metric_adapter_9b_20260922` | Completed, negative: frozen 9B metric head reduced held-out purity 0.3144 to 0.2903; test balanced accuracy 0.2445. | [Run](../code/fig6_v1/runs/metric_adapter_9b_20260922/) |
| 2026-09-22 | `fig5_v2_mri_forecast_final_20260922` | Completed: MRI transition figure and probability components; inference only. | [Run](../code/fig5_v2/runs/mri_forecast_final_20260922/) |
| 2026-09-22 | `metric_report_9b_20260922` | Completed: frozen 9B clinical metric head, validation-selected epoch 45; exploratory held-out plots and metrics. | [Run](../code/fig6_v1/runs/metric_report_9b_20260922/) |
| 2026-09-22 | `fig6_metric_report_9b_20260922` | Interrupted after comparison export; trained head preserved. Full figure restarted in fig6_metric_report_9b_v2_20260922. | [Run](../code/fig6_v1/runs/fig6_metric_report_9b_20260922/) |
| 2026-09-22 | `fig6_metric_report_9b_v2_20260922` | Completed: trained 9B image/report head figure; purity 0.4683, patient retrieval AUROC 0.5203; actual eMAR/ICU timelines and verified viewer. | [Run](../code/fig6_v1/runs/fig6_metric_report_9b_v2_20260922/) |
| 2026-09-22 | `paper_compact_9b_20260922` | Completed: image_report/report_tokens retrieval for 457 examinations; PNG/SVG/PDF and local HTML. | [Run](../code/fig6_v1/runs/paper_compact_9b_20260922/) |
| 2026-09-22 | `paper_compact_9b_v2_20260922` | Completed: image_report/report_tokens retrieval for 457 examinations; PNG/SVG/PDF and local HTML. | [Run](../code/fig6_v1/runs/paper_compact_9b_v2_20260922/) |
| 2026-09-23 | `paired_qwen35_9b_2gpu_48h_20260921_baseline` | Completed configured tests: classification, segmentation, vqa, segmentation_human. | [Run](../code/medworld/runs/paired_qwen35_9b_2gpu_48h_20260921/baseline/) |
| 2026-09-23 | `paired_qwen35_9b_2gpu_48h_20260921` | Completed: equal-step slots/no-slots training, both mask tests, classification/VQA and native Qwen9B comparison. | [Run](../code/medworld/runs/paired_qwen35_9b_2gpu_48h_20260921/) |
| 2026-09-23 | `raw_input_preflight_20260923` | Passed: raw baseline/slots three-task GPU smoke, common initialization, gradients, no baseline auxiliary branch and checkpoint reload; no full evaluation. | [Run](../code/medworld/runs/raw_input_preflight_20260923/) |
| 2026-09-23 | `current_only_preflight_20260923` | Passed: 138 tests plus 61 subtests; both arms three-task GPU checks and in-memory reload; no checkpoint files retained. | [Run](../code/medworld/runs/current_only_preflight_20260923/) |
| 2026-09-23 | `paired_qwen35_9b_2gpu_24h_20260923` | Interrupted by request before formal updates to implement Tables 1/2; both nine-update calibrations preserved. | [Run](../code/medworld/runs/paired_qwen35_9b_2gpu_24h_20260923/) |
| 2026-09-23 | `table12_preflight_20260923` | Passed: eight-task 9B/raw training, trained-weight reload, both native tables, official RadGraph, patient audits; 237 tests and 97 subtests. | [Run](../code/medworld/runs/table12_preflight_20260923/) |
