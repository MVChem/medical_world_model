# Cross-database MIMIC examples

This restricted local bundle contains one new single-interval CXR example, one
new four-state CXR trajectory, their same-patient MIMIC-IV clinical context,
and one independent MIMIC-III ICU reference episode.

## What is and is not linked

| Source pair | Relationship in this bundle |
|---|---|
| MIMIC-CXR-JPG v2.0.0 ↔ MIMIC-IV v3.1 | Patient/time linked by exact `subject_id`; every displayed CXR state lies in one unique admission and ICU stay. |
| MIMIC-III v1.4 ↔ MIMIC-IV v3.1 | No patient link. The III episode is independent because the public releases use regenerated IDs/date shifts and provide no crosswalk. |
| MIMIC-III v1.4 ↔ MIMIC-CXR-JPG v2.0.0 | No patient link. |

Accordingly, this directory uses real rows from all three databases, but it is
not a three-way same-patient trajectory. `concept_crosswalk.json` records only
exact ICU item-dictionary overlap and must not be interpreted as identity.

## Contents

- `single_interval/example_01`: two consecutive AP studies separated by about
  17.7 hours, with extracted reports and interval MIMIC-IV events.
- `multi_interval/example_01`: four consecutive AP studies with three adjacent
  intervals and cumulative horizons from state 0.
- `mimic_iii_reference/episode_01`: an independent 24-hour MetaVision ICU
  episode with non-cancelled, finished input and procedure records initiated in
  the fixed 24-hour window.
- `manifest.json`: CXR selection, paths, IDs, timestamps, hashes, and source
  transition JSONL line numbers; it also fingerprints the generator and CXR
  metadata table.
- `linkage_manifest.json`: CXR-to-IV joins and event counts.
- `clinical_context.json`: row-level IV data for each CXR example.
- `episode.json`: row-level III data for the independent episode.

Each source-table record includes the release, compressed CSV filename,
one-based decompressed CSV line number (header is line 1), and natural row key.
Images also include source/output SHA-256 checksums and the exact conversion.
`CHECKSUMS.sha256` covers every generated bundle file except itself.

In `linkage_manifest.json`, `event_counts` sums memberships across adjacent
segments. An event spanning a CXR boundary can therefore be counted in two
segments. `unique_source_row_counts` de-duplicates by source file and line.

## Time and treatment semantics

CXR pairs are exact-view, strictly adjacent studies in the full patient
timeline. For each adjacent interval, eMAR point events use `(t0, t1]`; ICU
inputs/procedures are included when `[starttime, endtime]` overlaps the open
interval. Longer state-0 horizons include their shown intermediate studies and
are not called adjacent.

The independent III episode instead selects records whose `STARTTIME` is in
`[window_start, window_end)`, then requires `CANCELREASON=0` and
`STATUSDESCRIPTION=FinishedRunning`. A procedure may continue past the window;
its temporal role is retained rather than being described as completed inside
the window.

Treatment files summarize events observed after the current image. They are
future information for a current-only forecast. If used as model input, the
task is a retrospective action-conditioned transition, not open forecasting.
These observational records do not establish treatment effects, and missing
rows do not prove that no treatment occurred. A device change described by a
radiology report is not claimed to be independently confirmed by an IV event
table unless a matching structured event is actually present.

## Rebuild

From `/home/data2/chk/workspace/2026`:

```bash
python 08/11/mimic_linkage/build_cross_database_examples.py
```

The generator reads the local credentialed releases under
`/home/data1/data/MIMIC` and overwrites only its named generated files.

## Access restriction

This directory contains MIMIC-derived images, reports, and clinical events.
Keep it inside an approved credentialed environment. Do not redistribute it as
an unrestricted artifact. See `MIMIC_LICENSE.txt`.
