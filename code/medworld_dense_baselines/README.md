# Frozen VLM dense baselines and visual-slot probes

Independent downstream probes for the six locally available Qwen3.5 / MedGemma
checkpoints. Results are recorded per run; manuscript tables are maintained in
[Table 1](../../27cvpr/tables/table1_future.tex) and
[Table 2](../../27cvpr/tables/table2_downstream.tex).

## September 13: frozen multidepth visual slots

This separate experiment extracts four fixed slots from different depths of each
checkpoint's native vision encoder, corresponding to visual slots 5–8. Only the
segmentation and ×4 SR decoders are trained, with a matched image-only control.
The [fixed protocol](https://github.com/MVChem/medical_world_model/blob/31c98ca14173620a3cec67da2defe559d05974da/research_notes/0913_frozen_multidepth_slots.md) defines
the representation, data, training budget, and supplementary shuffled-slot controls.

- Code: [extraction](frozen_slots_extract.py), [training](frozen_slots_train.py),
  [queue](frozen_slots_queue.py), and [reporting](frozen_slots_report.py).
- Run: `runs/frozen_slots_20260913/`; [report](runs/frozen_slots_20260913/REPORT.md),
  [queue status](runs/frozen_slots_20260913/status.json), and
  [saved protocol](runs/frozen_slots_20260913/protocol.json).

## September 12: frozen VLM dense baselines

This run is `runs/dense_20260912`. Read
[full_tables.md](runs/dense_20260912/preview/full_tables.md) for the original
Table 1/2 rows, completed zero-shot measurements, and trained-head rows.
[preview.md](runs/dense_20260912/preview/preview.md) documents the dense experiment
and queue in detail. The [experiment record](https://github.com/MVChem/medical_world_model/blob/31c98ca14173620a3cec67da2defe559d05974da/research_notes/0912_frozen_vlm_dense_baselines.md)
and the sections below describe this September 12 protocol.
Empty cells mean unmeasured/pending, never zero.

### Protocol

- VLMs are frozen. Extract final **language-model image-token** hidden states,
  preserving the native image token grid, and select 8 by 8 spatial bin centers
  and 1,024 uniformly spaced channel bin centers. Each model supplies `[64,1024]`.
  Matching shape does not imply semantic alignment or equal retained information.
- V-JEPA2.1 ViT-B EMA is frozen, image input 384, native spatial grid `[576,768]`.
- Segmentation and SR each have three input branches, all paired with VLM states:
  raw image; native V-JEPA with parameter-free normalization/channel subsampling;
  V-JEPA with a newly initialized, trained `768 -> 128 -> 64` adapter.
- The direct V-JEPA branch uses channels `6::12`, preserving all 24 by 24 spatial
  locations. All image branches feed a 64-channel map to the common decoder.
  Trainable parameter counts are recorded; adapter variants have additional
  parameters, while the same variant has exactly the same architecture and
  initialization for every VLM.
- SR uses 4x antialiased bicubic degradation. One uint8-rounded LR cache is shared
  by the VLM, V-JEPA, and raw-image head. HR is a target only. The raw-image SR
  baseline uses a standard bicubic LR residual. Feature replacements have **no
  pixel input and no pixel/bicubic skip**.
- Train each head independently for **20 epochs**, AdamW, initial LR 3e-4,
  cosine decay, weight decay .01, effective batch 8, seed 20260912. Identical
  epoch permutations across VLMs. Microbatch fallback preserves per-image loss
  weights and effective batch size. No early stopping or test-based selection.
- Save epoch 5/10/20 checkpoints and an optimizer checkpoint after every epoch.
  Final reported metrics use epoch 20. Validation is never used as a training set.
- Segmentation/SR: 4,096 train / 249 validation / 447 test frontal images, fixed
  from the existing patient split and CXAS quality mask. All six models and all
  three branches use identical IDs. Pseudo Dice averages right lung, left lung,
  and heart; this is teacher agreement, not human accuracy.
- Human Dice: 138 external Montgomery images, two lungs, **no heart label**.
  No Montgomery image enters training or model selection. NIH `leftMask` means
  image-left (anatomical right lung); `rightMask` means image-right. The coordinate
  convention is audited by mask centroids.
- Grounding pilot: Chest ImaGenome human **anatomical region** boxes, closed
  26-region query vocabulary, patient split 400/50/50 (train/validation/test),
  20,766 / 2,597 / 2,598 queries. This is **not MS-CXR lesion-phrase grounding**.
  MS-CXR was not locally available and official access is still credentialed.
  The matrix must retain this distinction if results are later used in a paper.
- Direction: 82 pairs / 154 finding-side fields from 284 annotated CIG gold pairs,
  after requiring unambiguous positivity at both ends, unambiguous direction,
  no duplicate overall/side fields, valid source, and 6h–30d horizon. Frozen VLM
  forecasts all six findings and all three scopes from source image/report plus
  horizon. Source report has a common 220-word cap; no future image/report or EHR
  is passed. This is a separately labeled direction forecast protocol, not a
  claim of evaluating the original 297 pairs. Only the gold persistent mask is
  scored; missing/invalid predictions remain false negatives. Macro F1 is over
  supported finding/scope/direction classes.

The task heads are independent: grounding training cannot change a VLM or the
segmentation/SR heads. Patient splits are disjoint **within each task**. Head
training does not establish absence of overlap with a backbone's pretraining.

### Metrics

Grounding uses normalized xyxy box IoU, mean IoU and fraction IoU >= .5.
Segmentation uses threshold .5, per-image/per-organ hard Dice on the valid ROI;
the test cohort is fixed before inference. SR uses mean per-image PSNR and
skimage SSIM (`data_range=1`, default uniform 7x7 window), full valid image ROI,
no border shaving. Dense confidence intervals use 1,000 patient-cluster bootstrap
samples. Every test record is saved; a missing record causes evaluation to fail.

### Running

Use `/home/data2/chk/workspace/2026/.venv/bin/python`. The installed Transformers
5.12.1 requires a compatible kernel loader for native FP8. The tested local
dependency is `kernels==0.12.3` in `vendor_compat/`, and the native Triton kernel is
`kernels-community/finegrained-fp8` version 2, snapshot
`061130fedf845f320c56de4425f7404f6512c87e`. Do not upgrade the shared environment.
The quantized Qwen checkpoint is executed natively in FP8, not dequantized to a
different precision variant. The MedGemma 27B checkpoint is **v1**, not 1.5.

```sh
python download_human.py
python prepare.py
python direction.py prepare
python test_contract.py
python coordinator.py --run /absolute/path/to/run --source /absolute/path/to/frozen/source
```

The coordinator uses only GPUs with no compute processes, memory below 512 MiB,
and utilization below 5%. It does not preempt other users. All eight GPUs are
eligible under the user's final instruction; GPU 4 is considered last. A single
run lock prevents duplicate coordinators. Work survives a chat disconnect,
resumes caches/epoch checkpoints, and retries failures up to three times. See
`queue.json`, `status.json`, and `logs/`; model-specific failures remain explicit.
The delivery target is **2026-09-14 08:00 Asia/Shanghai**.

Sources: [NIH Montgomery images/masks](https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/index.html),
[MS-CXR description and access](https://physionet.org/content/ms-cxr/1.1.0/),
[HF fine-grained FP8](https://huggingface.co/docs/transformers/quantization/finegrained_fp8).
