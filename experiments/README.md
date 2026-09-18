# Experiment history

Ongoing experiments: [registry.json](registry.json).
Details remain in the linked run directories. When a run ends, move its entry here with its date, outcome, and run link.

| Date | ID | Outcome | Run |
|---|---|---|---|
| 2026-09-18 | `featup_online_seed42_20260918` | Completed at 17:03 CST: all six variants reached 4,000 updates; validation, test, and human-test evaluations complete. | [Run and results](../code/medworld_spatial/runs/featup_online_seed42_20260918/) |
| 2026-09-18 | `patient_index_20260918` | Completed and validated: global MIMIC-IV/CXR index for 368,138 patients; 11.47 GiB cache. | [Run and results](../code/mimic_atlas/runs/patient_index_20260918/) |
| 2026-09-18 | `unpack_mimic_iv_20260918` | Completed: 31 CSVs (90.52 GiB); verified and removed gzip originals (9.92 GiB). Service checked. | [Run](../code/mimic_atlas/runs/unpack_mimic_iv_20260918/) |
| 2026-09-18 | `csv_offset_index_20260918` | Completed: 354 MiB offsets for 368,138 patients; React/FastAPI deployed, raw-data/browser checks passed, 11.47 GiB legacy copies removed. | [Run and results](../code/mimic_atlas/runs/patient_index_20260918_offsets/) |
