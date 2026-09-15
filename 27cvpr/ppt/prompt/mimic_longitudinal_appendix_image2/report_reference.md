# Report-derived reference notes

These concise paraphrases are provided only to help the image model understand why the cases were selected. They are not clinical adjudication, causal claims, or text that should appear in the figure. The follow-up reports and labels are observed targets/evaluation data rather than current-state model inputs.

## Case A — clear short-term change

- `t0`: Low lung volumes; no focal parenchymal opacity or pleural effusion was reported.
- `+2.1 d`: New extensive bilateral parenchymal opacities, greatest in the right upper lung, with possible left pleural effusion were reported.
- Linkage: one matched MIMIC-IV admission; TSICU at both studies.
- Transition ID: `mimiccxr_b69607cf13d5ee85cfa1`.

## Case B — stable control

- `t0`: Bibasal opacification was described as pleural effusions with compressive atelectasis.
- `+24.0 h`: Little interval change; the bibasal effusion/atelectasis pattern persisted.
- Linkage: one matched MIMIC-IV admission; SICU at both studies.
- Transition ID: `mimiccxr_b76df3bd6b77a134315d`.

## Case C — dense serial follow-up

- `t0`: Interstitial edema and a small right effusion; a CT-described right pneumothorax was poorly seen on the radiograph.
- `+9.2 h`: A right pigtail catheter had been placed; a small apical pneumothorax persisted and edema was improving.
- `+32.2 h`: The catheter had been revised or replaced; no appreciable pneumothorax was reported, with chronic fibrosis remaining.
- `+56.3 h`: No appreciable pneumothorax; chronic fibrosis and a likely small right effusion.
- Linkage: one matched MIMIC-IV admission; CCU followed by Vascular at the next three studies.
- Transition IDs: `mimiccxr_b6dbeb54ad552b93adfb`, `mimiccxr_5ea4a890c6b1684afcec`, `mimiccxr_4cbe6902bf9488125ef4`.

## Case D — long-horizon serial follow-up

- `t0`: Bilateral pleural effusions with atelectatic change; no new parenchymal opacity was reported.
- `+47.1 d`: Probable right lower-lobe collapse, no pneumonia, and small diminishing bilateral effusions.
- `+52.4 d`: New right upper-lung opacity reported as pneumonia, with increasing pleural effusions.
- Linkage: no single MIMIC-IV admission matched both `t0` and `+47.1 d`; the `+47.1 d` to `+52.4 d` segment matched one admission from ED to MICU/SICU.
- Transition IDs: `mimiccxr_5bb31183865a98df310c`, `mimiccxr_d96ab1c07165c38193e2`.

## Source boundary

- Images, reports, acquisition times, views, and report-derived observations: `MIMIC-CXR-JPG v2.0.0`.
- Admission matching and care-unit context: `MIMIC-IV v3.1`.
- An unmatched cross-episode segment means that no common admission satisfied the project linkage rule; it does not prove that no clinical encounter existed.
