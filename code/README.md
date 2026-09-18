# Code

- `code/` contains our code and the third-party repositories we use.
- All logs from our code go in `runs/`, organized by experiment or run name with a date suffix (`YYYYMMDD`), e.g. `runs/experiment_20260918/`.
- Third-party repositories keep their own structure and logging conventions.
- Recomputable intermediate results are computed on demand, never cached to disk. See [AGENTS.md](../AGENTS.md).
