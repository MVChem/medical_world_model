# Medical world-model code

The September Table 1 small-model experiment is in
[`medworld_table1/`](medworld_table1/README.md): Qwen3.5-0.8B, multimodal state
slots, latent future prediction, report/finding readouts, an 8h runner and
clinical evaluation. Its method follows `../research_notes/0907_paper_plan.md`.

Executable sources for the MIMIC VLA-JEPA baseline are consolidated here:

- `MIMIC_example/`: MIMIC-CXR transition builder, synthetic tests, and the
  restricted four-pair audit example;
- `mimic_vla_jepa/`: leakage-safe feature extraction, predictor training,
  evaluation, configurations, and tests; and
- `VLA-JEPA-reference/`: pinned official VLA-JEPA reference checkout with the
  documented PyTorch 2.11 BF16 RoPE compatibility patch.

Plans, checkpoints, and generated run metadata intentionally remain in
`../research_notes`, `../checkpoints`, and `../runs`. Downloaded upstream weights live in
`../checkpoints/base`; MIMIC-derived predictor weights live in
`../checkpoints/trained`.

From the `medical_world_model` project root, install both editable packages:

```bash
python -m pip install --no-deps -e code/VLA-JEPA-reference
python -m pip install --no-deps -e code
```

Then run the complete local test suites:

```bash
python -m unittest discover -s code/MIMIC_example/tests -v
python -m unittest discover -s code/mimic_vla_jepa/tests -v
```

See [`mimic_vla_jepa/README.md`](mimic_vla_jepa/README.md) for extraction,
training, resume, and evaluation commands.
