# Visual Slot Spatial Consistency (VSSC)

VSSC is a training objective that encourages the four visual slots to retain recoverable spatial information.
It draws on multi-view feature consistency, is not a reproduction of official FeatUp/JBU, and does not modify downstream task decoders.

```text
input image → encoder → 4 visual slots → unchanged task decoder → task loss
                              ↓
                   SlotSpatialReconstruction
                              ↓
                     64×64 feature field
                              ↓ same affine crop + average downsampling
input image → frozen native Qwen vision → feature targets → consistency loss
```

`visual_consistency.py` reconstructs features from slots. A 32×32 grid of coordinate queries reads four slots through cross-attention, followed by interpolation to 64×64 and convolutional refinement into 64 channels.
There is no image input, image stem, or image-guided kernel in the reconstruction branch. Sample-specific evidence must pass through the slots, and the auxiliary loss has no gradient path to the task decoder.

`targets.py` obtains targets from the frozen, unadapted Qwen vision branch: it reads patch tokens from the fourth selected depth, restores their spatial grid order, applies LayerNorm, and projects them to 64 channels using a fixed random orthogonal projection.
The teacher stays in eval/no_grad mode and shares immutable pretrained weights. It does not use online LoRA or EMA parameters.

Each sample uses the original view and two deterministic crops with scaling and translation. The teacher encodes each view independently.
Student features undergo the same coordinate transformations and average downsampling, then are compared with channel-normalized teacher features using squared distance within valid image regions.
Crops are determined by the seed, sample ID, and view index. Pixels and features are not cached.
For super-resolution, the teacher sees only LR inputs; HR images are used only for the original reconstruction supervision.

## Integration and controls

The default `visual_consistency_weight=0` creates no auxiliary modules.
`configs/qwen35_08b_vssc_2gpu.json` enables weight=0.1 and views=2 (three views including the original). The current two-GPU preset requests a combined 16-hour training budget through `run_experiment`, with per-rank batches of 16 (VQA: 8), accumulation of 1, and both baselines enabled.
This is an initial experimental weight, not a validated optimum. The SR task loss has a smaller scale, so monitor both raw and weighted auxiliary losses.

`model.py` assembles the modules and combines losses. The downstream task dispatcher additionally returns the already-computed state and does not own auxiliary modules.
The objective currently runs only during segmentation/sr updates when training and gradients are enabled, as the spatial current-task component of joint training.
It is not added to classification, report, or the temporal loss. The spatial current-task loss is task loss + weight × consistency; the temporal loss is added in the same optimizer update.
Logs record task loss, `visual_consistency`, and `visual_consistency_weighted` separately.

Evaluation and inference do not execute the auxiliary reconstruction branch or teacher; they use the same slots and standard task decoders.
Auxiliary trained parameters are retained in format-5 weights-only checkpoints.
Optimizer state is excluded, so these files do not support exact training resume.
The frozen target projection is reconstructed from a private seed.
The auxiliary reconstruction module uses a separate initialization RNG to preserve the main model initialization and sampling random stream.
Controlled experiments should use the same decoders, data, and training budgets, varying only the auxiliary objective. Such training comparisons have not yet been completed.
