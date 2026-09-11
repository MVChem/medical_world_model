# 2026-09-08 Table 1 pilot

Started at **2026-09-08 21:24 Asia/Shanghai** in tmux session
`medworld_pilot_20260908_8h`.

- [Live Table 1 and progress](runs/pilot_20260908_8h/table1.md)
- [Coordinator status](runs/pilot_20260908_8h/runner_status.json)
- [Main training status](runs/pilot_20260908_8h/ours/status.json)
- [Main training log](runs/pilot_20260908_8h/ours/console.log)
- [Native Qwen baseline log](runs/pilot_20260908_8h/direct/console.log)
- [Full protocol and limitations](README.md)

Qwen3.5-0.8B + V-JEPA 2.1 ViT-B, eight state slots, LoRA rank 8.
Main run: 1h task training + 7h future latent/report/finding training. The frozen
Stage-1 encoder control starts automatically at the first stage boundary.
Nominal checkpoint times are September 9 at approximately 01:24 and 05:24;
validation and I/O add wall time. Final clinical evaluation follows training.

Completed startup checks: four metric protocol tests, two model/gradient tests,
real-batch training for both architectures, Stage-1/midpoint/final checkpoint
save/load, matched control encoder freezing and update count, restart forward
loss agreement, and variable-length batched future generation. BF16 backward
numerics are not bitwise reproducible on restart; see README.

This file records launch settings. The live JSON/logs are authoritative for
current status. No completed 8h result is claimed at launch.
