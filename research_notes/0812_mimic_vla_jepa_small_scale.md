# MIMIC VLA-JEPA baseline plan

Date: 2026-08-12  
Status: engineering smoke test complete; scientific pilot is next  
Method: [VLA-JEPA](https://arxiv.org/abs/2602.10098) adapted to longitudinal chest X-rays

## Goal

Learn to predict the **representation of a future chest X-ray** from the
patient's current chest X-ray and current report.

This first baseline predicts a latent state. It does **not** generate a future
image or report, and it is not yet a clinical prognostic model.

## Input and output

| Role | Information |
|---|---|
| Model input | Current frontal CXR, current report when available, and a prespecified forecast horizon |
| Training target only | The observed future CXR from the same patient |
| Model output | Predicted future V-JEPA2 latent state |
| Never given as input | Future report, future labels, state change, or target identifiers |

The forecast horizon must be information we could provide at inference time,
such as “predict the state in the next few days.” The model should not silently
receive the exact follow-up interval derived from the target study.

## Model

```text
MODEL INPUTS
current CXR ──> frozen V-JEPA2 ──> current image state ───────┐
                                                              ├──> predictor
current CXR + current report + horizon                         │        │
            └──> Qwen3.5-4B ──> transition queries ───────────┘        v
                                                        predicted future state

TRAINING TARGET ONLY
future CXR ──> frozen V-JEPA2 ──> target future state

predicted future state <──────── latent training loss ────────> target future state
```

The predictor is necessary: it combines the current V-JEPA2 state with the
transition information produced by Qwen and predicts the future state. We use
the VLA-JEPA-style causal predictor rather than removing it.

Because a CXR is a single image, we repeat it into a short static clip for
V-JEPA2. Current and future CXRs are encoded separately so the noncausal video
encoder cannot leak future pixels into the current representation.

Primary training strategy:

- keep V-JEPA2 frozen;
- train the predictor;
- adapt Qwen with lightweight LoRA and trainable transition-query tokens; and
- use latent alignment between the prediction and the frozen future state as
  the training objective.

The robot action-generation head from VLA-JEPA is omitted because the current
MIMIC pairs do not contain actions or treatments.

## Data

Use adjacent frontal MIMIC-CXR studies from the same patient:

```text
current study -> next observed study
```

Keep the official patient-level train, validation, and test splits. The
training cohort must include stable, uncertain, and changed cases and must not
select examples using future labels or reports. Patients with many studies
should not dominate training.

The small curated example in `code/MIMIC_example/example_output` is only for
testing the pipeline. It is target-selected and cannot measure model quality.

All MIMIC-derived manifests, features, logs, and checkpoints remain on
restricted local storage.

## Work plan

### 1. Engineering baseline — complete

- Load Qwen3.5-4B and pretrained natural-image V-JEPA2.
- Build leakage-safe current/target data loading.
- Extract current, target, and Qwen bottleneck features.
- Train and resume the predictor on one or multiple GPUs.

This proved that the system runs, but the tiny smoke model mostly learned a
shortcut from the current image state and did not meaningfully use Qwen.

### 2. Unbiased pilot cohort — next

- Export a patient-safe training cohort from all eligible adjacent studies.
- Build validation data from the official validation split.
- Use broad, prespecified forecast horizons.
- Seal the official test split until design choices are fixed.

### 3. Primary training

- Freeze V-JEPA2.
- Train the predictor and Qwen LoRA/query tokens.
- Use distributed training across the available GPUs.
- Compare against models without Qwen, report text, or horizon information.

### 4. Evaluation

Start with tasks that do not require an image decoder:

1. **Future latent prediction:** is the predicted state closer to the future
   CXR than simply copying the current state?
2. **Future-image retrieval:** can the predicted state retrieve the correct
   future CXR from plausible alternatives?
3. **Future finding prediction:** does the predicted state contain information
   about future CheXpert findings?
4. **Transition prediction:** can it distinguish stable findings from findings
   that appear or disappear?

Evaluate separately across forecast horizons and patient groups. Report
patient-level uncertainty rather than treating repeated studies as independent.

### 5. Later extensions

Only after latent forecasting works:

- compare natural-image V-JEPA2 with a medical encoder;
- add MIMIC-IV treatments and events as observational context; and
- train a separate decoder or editor if future-image generation is desired.

## Success criteria

The pilot is useful only if it:

- beats the simple baseline of copying the current image state;
- performs better with the correct Qwen condition than with shuffled or zeroed
  Qwen queries;
- retains performance on unseen validation patients; and
- passes automated future-information leakage checks.

If these conditions fail, improve the cohort, target representation, or
bottleneck training before attempting image generation.

## Project locations

- Code and commands: [`code/mimic_vla_jepa/README.md`](../code/mimic_vla_jepa/README.md)
- MIMIC transition builder: [`code/MIMIC_example`](../code/MIMIC_example)
- Base and trained checkpoints: [`checkpoints`](../checkpoints)
- Run metrics and evaluations: [`runs`](../runs)
