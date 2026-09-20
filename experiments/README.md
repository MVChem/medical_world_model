# Experiment history

Ongoing experiments: [registry.json](registry.json).
Details remain in the linked run directories. When a run ends, move its entry here with its date, outcome, and run link.

| Date | ID | Outcome | Run |
|---|---|---|---|
| 2026-09-18 | `featup_online_seed42_20260918` | Completed at 17:03 CST: all six variants reached 4,000 updates; validation, test, and human-test evaluations complete. | [Run and results](../code/medworld_spatial/runs/featup_online_seed42_20260918/) |
| 2026-09-18 | `patient_index_20260918` | Completed and validated: global MIMIC-IV/CXR index for 368,138 patients; 11.47 GiB cache. | [Run and results](../code/mimic_atlas/runs/patient_index_20260918/) |
| 2026-09-18 | `unpack_mimic_iv_20260918` | Completed: 31 CSVs (90.52 GiB); verified and removed gzip originals (9.92 GiB). Service checked. | [Run](../code/mimic_atlas/runs/unpack_mimic_iv_20260918/) |
| 2026-09-18 | `csv_offset_index_20260918` | Completed: 354 MiB offsets for 368,138 patients; React/FastAPI deployed, raw-data/browser checks passed, 11.47 GiB legacy copies removed. | [Run and results](../code/mimic_atlas/runs/patient_index_20260918_offsets/) |
| 2026-09-19 | `qwen35_08b_joint_8h_20260919` | Completed: full held-out tests and fresh native Qwen3.5-0.8B comparison. | [Run](../code/medworld/runs/qwen35_08b_joint_8h_20260919/) |
| 2026-09-19 | `qwen35_08b_joint_8h_20260919_native_baseline` | Completed: fresh native Qwen3.5-0.8B tests and matched comparison; see parent COMPARISON.md. | [Run](../code/medworld/runs/qwen35_08b_joint_8h_20260919/qwen35_08b_joint_8h_20260919_native_baseline/) |
| 2026-09-20 | `three_tasks_smoke_20260920` | Passed: 36 tests; three-task real-data baseline/slots gradients and reload, two-sample inference, two-GPU training (replica spread 0), JSON testing disable verified. | [Run](../code/medworld/runs/three_tasks_smoke_20260920/) |
