# MIMIC longitudinal appendix — Image 2 asset pack

This folder contains a ready-to-use English prompt and the source radiographs for a four-case CVPR appendix figure. The selection deliberately covers a clear interval change, a stable control, a four-timepoint acute sequence, and a three-timepoint sequence with a 47.1-day gap.

## Use

1. Attach all eleven JPEGs in `images/` to Image 2.
2. Paste the contents of `PROMPT.txt`.
3. Keep the filename-to-case mapping intact if the interface renames uploads.
4. Use `report_reference.md` only as factual guidance; it is intentionally excluded from the requested on-canvas text.

The prompt gives Image 2 freedom over the overall layout and visual treatment. Its hard constraints are limited to scientific fidelity, source attribution, chronology, and the requested absence of a title or bottom explanatory copy.

## Files

- `PROMPT.txt`: the English generation prompt.
- `images/`: eleven source radiographs, renamed in chronological case order.
- `report_reference.md`: concise report-derived observations and linkage notes.

## Database provenance

- `MIMIC-CXR-JPG v2.0.0`: radiographs, radiology reports, study/image metadata, acquisition times, views, and report-derived labels.
- `MIMIC-IV v3.1`: admission matching and ICU/care-unit context only. This information is retrospective context and must not be presented as the source of the radiographs or as causal evidence.

The image files are local restricted MIMIC assets and are ignored by Git via this folder's `.gitignore`. Do not redistribute them or upload them to a service that is not permitted under your MIMIC data-use agreement.

The underlying curated records remain in `../../../../code/MIMIC_example/representative_output_10/transitions.jsonl` and `../../../../code/MIMIC_example/representative_linked_output_10/linked_transitions.jsonl`.
