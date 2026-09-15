# Table 1 comparator evaluation recovery, 2026-09-14

The original BioViL-T and CheXWorld runs completed all 2,400 Stage-2 optimizer updates. Their evaluations failed after generating the first 32 records because the adaptation wrapper called the shared prediction function without creating its output directory. The analogous missing parent directories in GREEN preparation were repaired as well.

This recovery uses the original configuration, source-only cohort, and final trained checkpoint. It does not retrain or change checkpoint weights. It writes to the previously nonexistent evaluation_test and green destinations expected by the original report. Existing frozen snapshots and failed attempt records remain unchanged.

The scheduler uses frozen Python sources with a SHA256 manifest and the existing shared UUID GPU locks. GPU 1 and GPU 2 are excluded. Coordinator PID and restart argv are in scheduler_recovery/launch.json; a coordinator restart uses that exact argv. Job evidence is in scheduler_recovery/attempts; scheduler_recovery/QUEUE.md and status.json expose progress. GREEN depends on successful clinical evaluation for each model.

Development fix: medworld_open_baselines/biovil_forecast.py. Requested plan: medworld_open_baselines/repair_forecast_plan_20260914.json. Clinical dependency smoke: medworld_open_baselines/repair_forecast_smoke_20260914/metric_environment.json (Transformers 4.57.1, RadGraph 0.1.18).
