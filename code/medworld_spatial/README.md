# Sparse spatial readout with FeatUp-inspired consistency

This isolated experiment asks whether a sparse patient state improves current
segmentation and ×4 SR, and exports the actual attentions used to make predictions.
The baseline unified implementation remains in `medworld/`, snapshotted at
`cc72e88`. Existing training runs and checkpoints are not modified.

## Model and supervision

- Start from the completed unified Qwen3.5-0.8B **Stage 1** checkpoint. Freeze
  Qwen/JEPA backbones and the four image-only fusion states; train the four visual
  slot queries, their readout projections, and a shared spatial decoder. This is
  slot/readout adaptation, not full VLM LoRA fine-tuning or a Stage 2 comparison.
- Pixel queries read eight slots at 32×32 and 64×64. A learned, image-guided
  3×3 neighborhood upsampler connects these levels. Local image features remain
  available to the decoder. The state is still eight 1,024-dimensional vectors.
- Following [FeatUp §3](https://arxiv.org/html/2403.10516v2), transform a predicted
  feature field with the same recorded crop/zoom as the image, downsample, and
  match frozen encoder features of the transformed image. Each requested batch
  computes identity plus two deterministic views on demand. Teacher projection is a fixed
  orthonormal 64-D map; the teacher cannot collapse with the student. The pilot
  uses fixed area downsampling and a different upsampler, so it is **not a FeatUp
  reproduction**. The fixed deterministic view set is an explicit speed/coverage tradeoff.
- Frozen native Qwen3.5-0.8B answers No/Yes for four fixed concepts on 2×2 image
  crops. Conditional No/Yes likelihoods are soft targets, not calibrated medical
  probabilities, anatomical masks, or native model attention. A concept is used
  only if crop probability spread is at least .08, maximum is at least .55, and
  all quadrants contain sufficient non-padding image support. These thresholds
  are fixed before evaluation, not fitted on test data.
- Text-only final language states initialize the inputs to a learned semantic
  query projection. Queries attend to the dense field, and the resulting concept
  summaries are routed back through those weights into the prediction features.
  Weak KL supervision compares attention mass in the four quadrants with VLM
  crop-score distributions. There is no assumption that raw text and visual
  hidden states already occupy the same dot-product space.
- Extra losses: `0.1 * feature consistency + 0.05 * semantic KL`. Original task
  losses remain BCE+Dice for segmentation and valid-pixel MSE for SR. SR scales
  the extra losses by **0.001** (effective weights 1e-4 / 5e-5), because its pixel
  MSE is about three orders of magnitude smaller than the segmentation loss.
  These coefficients are predeclared and shared with the image-only teacher
  control; they are not selected on test results. No new
  pixel/box annotations are introduced for alignment. Existing checkpoint
  training and dense task supervision are not described as unsupervised.

## Controls

All six variants share the same decoder architecture, initial weights for a given
seed, ordered batches, original task targets, and update count. Semantic modules
participate in all forward paths; their extra weak loss is enabled only below.

| Variant | State condition | Feature consistency | Weak semantic KL |
|---|---|---|---|
| `image_only` | All state values zero | No | No |
| `visual_slots` | Four visual slots; fusion positions zero | No | No |
| `slots` | Four frozen fusion + four trainable visual slots | No | No |
| `featup` | Same eight slots | Yes | No |
| `featup_semantic` | Same eight slots | Yes | Yes |
| `image_only_featup` | All state values zero | Yes | Yes |

The last control separates benefits of the teacher losses/image decoder from
benefits of the sparse state. Adding JEPA features is tested through the fusion
slots, not an extra dense JEPA bypass. Zero-slot inference is also exported as a
sensitivity diagnostic; it does not replace matched retraining.

## Data and evaluation

Use `UnifiedData`'s existing global patient holdouts. Select training/validation
subsets by a predeclared hash of image ID, never by performance. The overnight
plan requests up to 4,096 training images per task, 128 validation images, the
full available 447-image test selection and 138-image external human lung set.
Counts can be lower after the unified holdout filter; each group’s `protocol.json` records the actual counts.
Original JPG/PNG files are decoded for every batch using the recorded padding
boxes. Segmentation uses a 256-square input canvas. SR inputs are computed by
antialiased bicubic downsampling to 128 square, with the original uint8 rounding.
Every SR encoder, teacher, semantic crop and decoder sees only this LR image; HR
is only the 512-square task target. Reports are absent from encoder inputs.
Original CXAS supervision is read as fixed task labels; Montgomery masks are
decoded from the original PNGs. No regenerated image arrays are written.
`verify_sources` streams all 5,830 reconstructed inputs and 138 human masks into
the original array hashes without saving the arrays.

Tasks alternate at batch 8. The online run defaults to seed 42 and 4,000
updates (2,000 per task); `plan.json` is authoritative for each run’s step budget. All conditions use AdamW, learning rate 1e-4, weight decay
.01, gradient clipping 1, and BF16 autocast. Checkpoints are evaluated at fixed
updates; no test-based selection. The queue reserves up to the last 30 minutes for
evaluation and records truncated budgets explicitly. Seed deltas appear only
for conditions completing the same budget. A small gain without repeatability
is not interpreted as evidence of improved spatial representation.

## On-demand execution

There is no preparation phase and no feature, slot, transformed-view, or semantic
score cache on disk. One worker per seed loads the frozen teachers and six
independent matched models. It computes the current batch once, updates all six
models with that same batch, then discards the intermediate tensors. Only model
weights, fixed projection/text-query initialization and row metadata persist in
memory. Repeated examples are decoded and recomputed.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=code \
  /home/data2/chk/workspace/2026/.venv/bin/python -m medworld_spatial.night \
  --checkpoint code/medworld/runs/qwen35_08b_2gpu_day_gpu67_20260916_161228/stage1.pt \
  --semantic-teacher code/medworld_table1/weights/Qwen3.5-0.8B \
  --out code/medworld_spatial/runs/featup_online_20260918 --hours 8 --seeds 42 --steps 4000
```

The launcher saves the actual source snapshot and hashes, including uncommitted
changes, then starts an independent **systemd user service**. `launch.json`
records its unit, PID and deadline. The service records worker start/exit events,
return codes and failures. It survives the launching shell/session. GPU preference
is **1, 2, 3, 6, 7, 0**, only when idle and unlocked; **4 and 5 are forbidden**.
Other processes are never terminated.

`runs/active.json` points to the current run. Each `groups/seed*/last.pt` atomically
saves all six models and optimizers at the same completed update. Resume checks
the input/teacher fingerprints and experiment parameters. `jobs/*/` holds each
variant’s losses, gradient audit, final metrics and attention exports. These final
outputs and checkpoints are retained; they are not input caches for another run.
The retired `medworld_spatial.prepare` command refuses to create disk caches.

Eight hours are measured from the new launch. Online teacher inference is
included. Budget-truncated training and partial evaluations are labelled, and
matched contrasts require equal completed budgets and complete evaluations.

```bash
PYTHONPATH=code /home/data2/chk/workspace/2026/.venv/bin/python \
  -m medworld_spatial.report --run code/medworld_spatial/runs/featup_online_20260918 --render
```

## Attention exports

Raw arrays, provenance, and common-scale previews distinguish:

1. Encoder visual-slot → native image-patch attention, with Qwen's merge-block
   token ordering explicitly inverted into raster coordinates.
2. Decoder pixel-query → eight-slot attention at both spatial scales; maps
   average heads, raw probabilities sum to one over slots per pixel.
3. Semantic phrase-query → feature-field attention, with weights also used to
   route concept summaries into predictions; sums are over valid pixels.
4. Local 3×3 upsampling weights, used directly in feature aggregation.

Encoder/phrase heatmaps display weight relative to uniform spatial attention;
each panel group has a shared color scale. Decoder maps display raw slot
weights. Interpolation is only for display, and raw resolutions are saved.
Fixed hash-selected validation cases show predictions and errors alongside
attention; maps are neither lesion annotations nor proof of clinical causality.

## Validation

```bash
PYTHONPATH=code CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 \
  /home/data2/chk/workspace/2026/.venv/bin/python \
  -m pytest code/medworld/tests code/medworld_spatial/tests -q -p no:cacheprovider
```

Tests cover patch order, view alignment, fixed teacher gradients, weak-target
filtering, attention normalization/padding, task gradients, the image-only
control, and SR's forward independence from HR targets. Real-data smoke runs
also check on-demand slots against the original model and exact checkpoint reload.
Tests cover matched-group optimizer resume, stateless batch order and forbidden GPUs.
