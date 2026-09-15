# MIMIC-CXR data appendix

Each state is one AP radiograph plus the same-study report text:

- `.png`: full radiograph, converted without cropping or resizing
- same-stem `.txt`: extracted `FINDINGS` and/or `IMPRESSION`

## Single-interval pairs

| Folder | Pair | Elapsed time (rounded) |
|---|---|---:|
| `single_interval/example_01` | current → future | 20.1 h |
| `single_interval/example_02` | current → future | 24.0 h |
| `single_interval/example_03` | current → future | 50.5 h |

These are same-patient, matched-AP, strictly adjacent studies.

## Multi-interval examples

| Folder | Pairings from state 0 | Cumulative elapsed times (rounded) |
|---|---|---|
| `multi_interval/example_01` | state 0 → states 1, 2, 3 | +30.5 h, +66.6 h, +77.6 h |
| `multi_interval/example_02` | state 0 → states 1, 2, 3 | +47.5 h, +71.7 h, +95.8 h |
| `multi_interval/example_03` | state 0 → states 1, 2, 3 | +20.4 h, +42.3 h, +50.1 h |

Each folder is an ordered four-study trajectory. Pair `state_0_current` with each `state_[1-3]_future`. Consecutive states are adjacent studies; the state 0 → state 2/3 pairings are longer-horizon views that explicitly include the intervening states.

Future images and reports are observed targets/evaluation evidence, not forecasting inputs. Exact elapsed times here are appendix metadata; the model contract uses coarse horizon bins. See `manifest.json` for study-level provenance.

## Linked clinical context

Each example also has:

- `clinical_context.json`: traceable MIMIC-IV v3.1 admission, ICU stay,
  eMAR, ICU input, and ICU procedure rows
- `treatment_prompt.txt`: a short deterministic summary of the retrospectively
  observed interval events

The CXR-to-IV join uses the shared `subject_id` and aligned shifted timestamps.
Every attached IV row records its source table, compressed-file line number, and
row key; `linkage_manifest.json` summarizes all six joins. MIMIC-III is not
attached to these patients because its public identifiers and date shifts cannot
be crosswalked to MIMIC-IV; the manifest records that audit explicitly.

Treatment context after the current CXR is not available to a current-only
forecast. Supplying it defines the separate
`retrospective_action_conditioned_transition` task and does not establish a
causal treatment effect.

Restricted MIMIC data: keep these files in an approved environment and do not redistribute them as unrestricted assets. See `MIMIC_LICENSE.txt`.
