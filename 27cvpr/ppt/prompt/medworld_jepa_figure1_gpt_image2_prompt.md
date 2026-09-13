# Figure 1 design brief — MedWorld-JEPA

The canonical editable figure is
`imgs/medworld_jepa_architecture.svg`; the generated manuscript asset is
`imgs/medworld_jepa_architecture.pdf`. This brief may be used to create
alternative visual drafts, but any replacement must preserve the scientific
content below.

## Core message

Create a publication-ready, flat vector-style methods figure for a two-column
CVPR paper. The figure must explain one distinction:

> State tokens describe what the patient contains at the current time; task
> queries specify what the model should do with that state.

The model learns a horizon-independent current patient state from information
available at or before time (t). Authentic longitudinal pairs shape that
current state through fine-grained pathology and lesion forecasting. Future
images and reports are training targets only.

## Required layout

Use a 2:1 landscape canvas with four visually distinct regions.

### 1. Current branch

Show:

- current XR, CT, or MRI study;
- current report and permitted history, all timestamped no later than (t);
- one trainable modality-aware 2D/3D JEPA encoder;
- multi-scale dense features;
- a fixed-size structured current state with exactly five labeled token groups:
  global, anatomy, pathology, lesion, and trajectory.

Label the state as horizon-independent. The trajectory token belongs to the
current state: it reads current and permitted past information, not the future,
and it does not receive a forecast horizon.

### 2. Task queries

Show four queries and their selective read routes:

- `CLS` reads global + pathology + trajectory;
- `SEG` reads dense + anatomy + lesion;
- `RETRIEVAL` reads global + pathology;
- `FORECAST(h)` reads pathology + lesion + trajectory.

Only `FORECAST(h)` receives the horizon. Queries read the state and never
rewrite it.

### 3. Objective routing

Show four direct supervision routes:

- volume JEPA → dense + anatomy;
- current report/finding alignment → pathology;
- boxes, masks, region text, or weak localization → lesion;
- true patient pairs/triples → trajectory and fine-grained future prediction.

Make clear that token roles arise from input visibility, attention
connectivity, and group-specific losses, not from token names alone.

### 4. Target-side branch

Use a dashed green training-only container. Show the observed future study and
optional future report entering a stop-gradient target path. Its learned visual
and state encoders are EMA copies; a fixed text encoder or frozen label parser
may also construct targets. The target consists of future pathology and lesion
slots, the next trajectory condition used for rollout, and clinically defined
transition events.

Forecast outputs must be fine-grained:

- pathology: onset, persistence, improvement, worsening, resolution;
- lesion: birth, death, growth, shrinkage, morphology change.

Illustrate permutation-invariant lesion matching or birth/death slots without
showing a generated future image. No future input may enter the online current
branch.

## Style

Use a restrained color-blind-safe palette:

- blue for image geometry, dense features, and anatomy;
- magenta/purple for pathology and trajectory;
- orange for lesion and task queries;
- green only for EMA targets and supervision;
- charcoal for inputs, labels, and inference arrows.

Use flat pale fills, crisp sans-serif type, modest rounded corners, and
orthogonal connectors. Solid charcoal arrows denote online data flow; dashed
green arrows denote target-only supervision. Keep every label readable at
two-column print size.

## Hard constraints

- Do not show a generic pooled multimodal state as the reusable interface.
- Do not show a horizon-conditioned transition token as part of the current
  state.
- Do not show a Gaussian-mixture head, future A/G/M prediction, optional
  intervention conditioning, VQA, or treatment counterfactuals.
- Do not show frozen V-JEPA2, Qwen vision, VLA-JEPA, 24 learned queries, or an
  L1-only objective.
- Do not reconstruct pixels or voxels and do not render a generated future
  scan.
- Do not imply that future supervision improves every downstream task.
- Do not let a future image, report, token, or label enter the online branch.
- Do not add numerical results, citations, decorative anatomy, logos,
  watermarks, gradients, shadows, or stock imagery.
