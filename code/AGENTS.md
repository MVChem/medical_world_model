# Development Guidelines

- Maintain a `.gitignore` in each project's folder. Track source code, configuration, and documentation; exclude data, logs, model weights, checkpoints, and generated outputs.
- Write all logs from our code to `runs/`, using experiment or run names with a `YYYYMMDD` date suffix, e.g. `runs/experiment_20260918/`.
- Preserve third-party repositories' structure and logging conventions.
- Compute recomputable intermediate results on demand; never cache them to disk.
