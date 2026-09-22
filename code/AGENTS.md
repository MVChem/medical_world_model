# Development Guidelines

- Maintain a `.gitignore` in each project's folder. Track source code, configuration, and documentation; exclude data, logs, model weights, checkpoints, and generated outputs.
- Write all logs from our code to `runs/`, using experiment or run names with a `YYYYMMDD` date suffix, e.g. `runs/experiment_20260918/`.
- Preserve third-party repositories' structure and logging conventions.
- Compute recomputable intermediate results on demand; never cache them to disk.
- Store downloaded datasets under `/home/data2/chk/data`; expose datasets under `code/data` only through symbolic links. Keep prepared MedWorld manifests in `/home/data2/chk/data/medical_world_model/` with the same project-link convention.

## Data presentation

- Prefer modern, interactive HTML interfaces when presenting or exploring data. Make the available data easy to browse, filter, and inspect.
- Use React, FastAPI, or similar tools when they suit the project; simple HTML/CSS/JavaScript is also welcome.
- Look for polished design references in Figma or other websites before designing the interface. Adapt their layout, typography, colors, and interactions to the data, and record the reference links in the project's documentation.
