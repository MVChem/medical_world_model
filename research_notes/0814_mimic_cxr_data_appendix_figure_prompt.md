# MIMIC-CXR Data Appendix Figure Prompt

Attach these six files in order, then paste the prompt below into Image 2:

1. `assets/mimic_data_appendix/single_interval/example_01/state_0_current.png`
2. `assets/mimic_data_appendix/single_interval/example_01/state_1_future.png`
3. `assets/mimic_data_appendix/single_interval/example_02/state_0_current.png`
4. `assets/mimic_data_appendix/single_interval/example_02/state_1_future.png`
5. `assets/mimic_data_appendix/single_interval/example_03/state_0_current.png`
6. `assets/mimic_data_appendix/single_interval/example_03/state_1_future.png`

Pairing details and the separate multi-interval examples are in [`assets/mimic_data_appendix/README.md`](assets/mimic_data_appendix/README.md).

```text
Use case: scientific-educational
Asset type: full-width 16:9 appendix figure

Create a publication-ready figure titled “Longitudinal MIMIC-CXR Pairs for Latent Forecasting.” Use three rows (A–C) with columns “CURRENT CXR (.png)” | “PAIRED RADIOLOGY REPORT (.txt; ABRIDGED)” | “COARSE HORIZON” | “OBSERVED FOLLOW-UP CXR.” Bracket columns 1–3 as “MODEL INPUT” and column 4 as “TARGET ONLY.” Make the report column widest; use readable white cards headed “REPORT EXCERPT,” never placeholder lines.

A: Images 1–2; “0–24 h”; “No definite vascular congestion, pleural effusion, acute pneumonia, or pneumothorax.” B: Images 3–4; “0–24 h”; “Increased opacification at both bases, consistent with pleural effusion and compressive atelectasis.” C: Images 5–6; “24–72 h”; “There is no focal parenchymal opacity suggesting pneumonia or aspiration. No evidence of pneumothorax. No pleural effusions.” Render excerpts verbatim.

Preserve each radiograph exactly and show its full field of view; no crop, redraw, retouching, or synthetic anatomy. Label arrows “same patient · adjacent study · matched AP view.” Under the target header write “Future report + CheXpert labels: evaluation only.” Footer: “Illustrative examples only; observed follow-up is one retrospective outcome.”

Style: white background; blue-gray input band, muted green target band, amber horizon pills; thin rules, compact sans-serif, generous whitespace. No architecture, callouts, icons, logos, or watermark.
```

## Suggested caption

**Illustrative longitudinal MIMIC-CXR pairs.** The model receives a current frontal radiograph, its current report, and a prespecified coarse horizon; the adjacent same-view follow-up radiograph supplies the latent-space training target. Future reports, labels, and exact realized intervals are excluded from model input. These manually selected examples illustrate the data contract and are not representative cohort samples.
