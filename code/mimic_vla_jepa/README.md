# MIMIC VLA-JEPA baseline

This package implements the first baseline in
[`https://github.com/MVChem/medical_world_model/blob/31c98ca14173620a3cec67da2defe559d05974da/research_notes/0812_mimic_vla_jepa_small_scale.md`](https://github.com/MVChem/medical_world_model/blob/31c98ca14173620a3cec67da2defe559d05974da/research_notes/0812_mimic_vla_jepa_small_scale.md):

```text
current frontal CXR + current report + prespecified coarse horizon bin
  -> frozen Qwen3.5-4B latent query states
  -> released VLA-JEPA action-conditioned predictor
  -> frozen future V-JEPA2 state
```

The future image is used only by the frozen target encoder. Future reports,
target states, and state deltas never enter Qwen. The current implementation
trains the predictor from locally cached restricted-data features; it does not
train an image decoder or make a clinical forecasting claim.

## Environment

The shared environment already contains PyTorch, Transformers, PEFT,
Safetensors, and timm. The released predictor is imported from official
VLA-JEPA commit `ec8c70f6e155e2377bbd4d787004c14179c00c7c`:

```bash
cd /home/data2/chk/workspace/2026/08/04/medical_world_model
python -m pip install --no-deps -e code/VLA-JEPA-reference
python -m pip install --no-deps -e code
```

That checkout has one local PyTorch 2.11 compatibility patch: the rotary
query/key result is cast back to its input dtype before SDPA. Without it,
float32 rotary positions promote BF16 query/key tensors while values remain
BF16, and SDPA rejects the mixed dtypes. The patch does not change the model
shape or loss.

Place the two ungated base checkpoints under local restricted storage. The
smoke run uses:

- `Qwen/Qwen3.5-4B`
- `facebook/vjepa2-vitl-fpc64-256`

The downloaded revisions used by the smoke artifacts are respectively
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` and
`b3c1679b7c34d3255ef3547f27c7b226aefab26f`.

Do not upload feature caches, checkpoints, prompts, or MIMIC-derived logs.

## Tests

Run both suites from `medical_world_model`:

```bash
python -m unittest discover -s code/mimic_atlas/tests -v
python -m unittest discover -s code/mimic_vla_jepa/tests -v
```

## Extract the four restricted smoke features

Single GPU:

```bash
python -m mimic_vla_jepa.extract_features \
  --manifest code/mimic_atlas/runs/exports/legacy_20260918/example_output/forecast_manifest.jsonl \
  --output-dir runs/features_smoke \
  --qwen-model checkpoints/base/Qwen3.5-4B \
  --vjepa-model checkpoints/base/vjepa2-vitl-fpc64-256 \
  --max-records 4
```

Four independent GPU ranks (one pair per rank):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
  --standalone --nproc-per-node=4 \
  -m mimic_vla_jepa.extract_features \
  --manifest code/mimic_atlas/runs/exports/legacy_20260918/example_output/forecast_manifest.jsonl \
  --output-dir runs/features_smoke \
  --qwen-model checkpoints/base/Qwen3.5-4B \
  --vjepa-model checkpoints/base/vjepa2-vitl-fpc64-256 \
  --max-records 4
```

Every CXR is repeated for eight V-JEPA frames, then its four tubelet states are
averaged. Current and future CXRs are encoded in separate calls so the
noncausal video encoder cannot leak future information into the current state.

New feature extraction rebuilds the prompt from the current report and one of
four shared coarse bins: `0-24h`, `24-72h`, `3-7d`, or `>7d`. It never reuses
the combined manifest's serialized `prompt`, and the exact realized interval is
not rendered. Legacy manifests without `horizon_bin` remain readable: the bin
is derived from `elapsed_hours`, but the resulting prompt is still coarse.
Explicit bins inconsistent with `elapsed_hours` are rejected. Historical
four-case cached features predate this contract and must not be mixed with a
new serious cohort.

Build leakage-safe train/validation manifests with the commands in
[`../mimic_atlas/README.md`](../mimic_atlas/README.md), then extract each
split to a separate restricted feature directory. Do not take the first N rows
from a `change_enriched` manifest; deterministic sampling must happen in the
builder before extraction.

## Train the micro predictor

One-GPU correctness run:

```bash
CUDA_VISIBLE_DEVICES=0 python -m mimic_vla_jepa.train_predictor \
  --features runs/features_smoke/features.jsonl \
  --config code/mimic_vla_jepa/configs/micro.yaml \
  --run-dir runs/predictor_smoke_1gpu
```

Eight-GPU DDP smoke run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run \
  --standalone --nproc-per-node=8 \
  -m mimic_vla_jepa.train_predictor \
  --features runs/features_smoke/features.jsonl \
  --config code/mimic_vla_jepa/configs/micro.yaml \
  --run-dir runs/predictor_smoke_8gpu
```

With four cached examples and eight ranks, the distributed sampler repeats
examples to give each rank one item. That run validates DDP only; it is not an
efficiency or quality measurement.

Version-2 and version-3 checkpoints store the sampler epoch/batch position, one
RNG state per rank, feature-manifest SHA-256, world size, and resolved configs. Resume refuses
changed immutable inputs. In the recorded eight-GPU check, a step-5 checkpoint
resumed through step 7 had identical logged loss/gradient metrics to an
uninterrupted seven-step run; final model tensors differed by at most
`9.31e-10`. Legacy version-1 smoke checkpoints remain evaluable but are not
accepted for reproducible resume.

Evaluate a checkpoint against persistence and query ablations:

```bash
CUDA_VISIBLE_DEVICES=0 python -m mimic_vla_jepa.evaluate_predictor \
  --features runs/features_smoke/features.jsonl \
  --checkpoint checkpoints/trained/predictor_overfit_4cases_1gpu/checkpoint_last.pt \
  --output runs/predictor_overfit_4cases_1gpu/evaluation.json
```

For a wall-clock-bounded cached-feature run, use the 24-hour config together
with separate patient-level training and validation feature manifests:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
  --standalone --nproc-per-node=4 \
  -m mimic_vla_jepa.train_predictor \
  --features /restricted/features_train/features.jsonl \
  --split train \
  --eval-features /restricted/features_validate/features.jsonl \
  --eval-split validate \
  --config code/mimic_vla_jepa/configs/serious_24h_cached.yaml \
  --run-dir runs/predictor_cached_24h_4gpu
```

That config stops after 24 cumulative hours and checkpoints/evaluates at
approximately 8-hour wall-clock boundaries. A baseline validation pass runs
before the first update and a final pass runs at shutdown. Distributed
validation assigns every example to exactly one rank and records prediction,
copy-state, zero-query, and shuffled-query metrics in `evaluations.jsonl`.
Version-3 checkpoints preserve cumulative elapsed time; version-2 checkpoints
remain resumable and begin their elapsed clock at zero because they did not
record prior wall time. This command still trains only from frozen cached Qwen
queries; it is not the online-Qwen-LoRA training path.

## Train online Qwen LoRA and learned queries

The serious path caches only frozen V-JEPA source/target states. Qwen runs
online from the current image, current report, and prespecified coarse horizon;
its text-tower LoRA adapters, 24 independent query embeddings, and the full
VLA-JEPA predictor are optimized jointly. The Qwen vision tower and all base
weights remain frozen. Training and validation use independent patient-level
manifests and independent state caches.

Extract the two state-only caches first:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
  --standalone --nproc-per-node=4 \
  -m mimic_vla_jepa.extract_states \
  --manifest /restricted/cohort_train/forecast_manifest.jsonl \
  --output-dir /restricted/states_train \
  --vjepa-model checkpoints/base/vjepa2-vitl-fpc64-256 \
  --split train

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
  --standalone --nproc-per-node=4 \
  -m mimic_vla_jepa.extract_states \
  --manifest /restricted/cohort_validate/forecast_manifest.jsonl \
  --output-dir /restricted/states_validate \
  --vjepa-model checkpoints/base/vjepa2-vitl-fpc64-256 \
  --split validate
```

Before the long run, exercise the exact full-size model and four-GPU DDP path
for one optimizer step. This includes an initial four-example validation and a
final validation/checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
  --standalone --nproc-per-node=4 \
  -m mimic_vla_jepa.train_serious \
  --forecast-manifest /restricted/cohort_train/forecast_manifest.jsonl \
  --state-features /restricted/states_train/states.jsonl \
  --eval-forecast-manifest /restricted/cohort_validate/forecast_manifest.jsonl \
  --eval-state-features /restricted/states_validate/states.jsonl \
  --qwen-model checkpoints/base/Qwen3.5-4B \
  --config code/mimic_vla_jepa/configs/serious_smoke_4gpu.yaml \
  --run-dir runs/serious_online_smoke_4gpu
```

After that smoke succeeds, launch the 24-hour configuration with the same four
inputs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
  --standalone --nproc-per-node=4 \
  -m mimic_vla_jepa.train_serious \
  --forecast-manifest /restricted/cohort_train/forecast_manifest.jsonl \
  --state-features /restricted/states_train/states.jsonl \
  --eval-forecast-manifest /restricted/cohort_validate/forecast_manifest.jsonl \
  --eval-state-features /restricted/states_validate/states.jsonl \
  --qwen-model checkpoints/base/Qwen3.5-4B \
  --config code/mimic_vla_jepa/configs/serious_4gpu.yaml \
  --run-dir runs/serious_online_24h_4gpu
```

The online configuration has a cumulative 24-hour active-training wall-clock
guard and runs evaluation/checkpoint events at approximately 8-hour training
boundaries. Required evaluation and checkpoint I/O pause this clock, and final
evaluation plus the final save deliberately run after the training cutoff, so
the tmux process can finish later than 24 elapsed real hours. A terminal step
does not also write a redundant scheduled checkpoint. Learning rates use a 3%
wall-clock warmup followed by cosine decay to 10% of each component's base
rate; scheduler state and cumulative time are resumed.

The full validation set is evaluated at startup, at intervals, and at
shutdown. Metrics include prediction, copy-state, zero-query, and shuffled-query
conditions. The shuffle always uses a different validation patient even when
per-rank batch size is one. Long reports are shortened within a fixed prompt
token budget before chat construction; automatic sequence truncation is
disabled, preserving the image placeholders and all 24 query markers.

Serious checkpoints contain the LoRA adapters, learned query embeddings,
predictor, optimizer, scheduler, per-rank RNG states, sampler position, and
resolved config. Immutable identity covers SHA-256 hashes for all four raw/state
manifests, both state-cache metadata files, Qwen weight shards, Qwen config,
tokenizer, chat template, and image processor configuration. Referenced CXR
and state tensor files are assumed read-only for the run; they are not hashed
one by one. Resume requires the same four-rank topology and identities, and
truncates log/evaluation artifacts that are newer than the selected checkpoint.

Training metrics and run metadata are written under `runs/<run-name>/`.
Checkpoint weights default to `checkpoints/trained/<run-name>/`; pass
`--checkpoint-dir` only to select another restricted checkpoint location.

Summarize either an active or completed run without loading its multi-gigabyte
checkpoint:

```bash
python -m mimic_vla_jepa.summarize_serious \
  --run-dir runs/serious_online_24h_4gpu
```

Add `--require-complete` in automated audits. It exits nonzero until the
0/8/16/24-hour evaluations, two periodic checkpoints, final checkpoint, and
24-hour result are all present. The summary reports validation deltas against
the initial model and copy-current baseline, query-ablation effects, training
throughput, finite metric checks, and milestone status.

The released-size adapted predictor config is
`code/mimic_vla_jepa/configs/full_size_adapted.yaml`: 12 layers, 8 heads, width 1024, and 24 query
positions. The 24-position single-transition construction preserves the
released total query budget but is not literal paper trajectory parity.

## Recorded smoke outcome

The eight-GPU run completed seven updates including a checkpoint resume. A
100-step four-case overfit reduced aggregate L1 from the copy baseline's
`1.2406` to `1.0393`, but shuffled-query L1 was `1.0394` and zero-query L1 was
`1.0416`. The predictor therefore learned a source-state shortcut and did not
demonstrate meaningful use of the frozen Qwen bottleneck. These curated cases
are only a wiring test; the next experiment is an unbiased cohort with online
Qwen LoRA/trainable query tokens.

New restricted output directories are created mode `0700`, and atomic feature,
metadata, and checkpoint files are mode `0600`.
