# Qwen3.5-9B paired-image/report screening

This is a retrospective semantic audit. Each request contains **two actual
chest radiographs plus both untruncated cleaned Findings/Impression sections**.
No CheXpert labels, curator labels, treatments, or downstream outcomes are
provided to the VLM. All patient data stays on localhost.

## Denominators

MIMIC-CXR contains 377,110 images in 227,835 studies. 243,334 images are AP/PA.
The original 70,318 adjacent-study pairs use 99,695 unique images; this is a
pair-design subset, not an estimate of the number of usable images in MIMIC.
32,695 patients have only one study and cannot supply a longitudinal pair.

`audit_pair_funnel.py` reconstructs the sequential exclusions and sensitivity
counts. Allowing AP↔PA and all positive gaps gives 136,385 adjacent-study pairs,
170,668 unique images, 170,632 studies and 29,897 patients. Of these, 70,591 pairs
have a unique shared admission. All 70,318 original pair IDs are retained
exactly through `original_strict_subset`; original cohort tiers remain attached.

The expanded manifest still requires an AP/PA image at both endpoints and parsed
Findings/Impression sections. 15,294 otherwise ordered frontal candidate pairs
have unsupported report formatting at one or both endpoints; they need separate
review, and should not be called missing reports or bad images. This manifest
does not exhaust single-study training data, lateral image use, or all possible
nonadjacent pair combinations. It does not skip an intervening study to manufacture
an adjacent pair. No images are deleted.

## Labels and limits

`primary_class`: `changed`, `stable`, `indeterminate`.

`direction`: improved, worsened, mixed, other_change, uncertain, not_applicable.
`stable_state` distinguishes persistent abnormalities from no acute findings.
Device changes, technical confounds, image/report assessments and short evidence
strings are separate fields. Stable does not mean healthy; unknown or failed
requests never count as stable. Image assessment is report-conditioned, **not**
a blinded image-only experiment. “Unchanged” in a report might refer to another
prior study; the prompt explicitly instructs comparison of the supplied A/B.

Labels are model-generated weak labels, not radiologist truth or a model accuracy
evaluation. Qualitative confidence is not calibrated. Field contradictions and
image/report disagreement are flagged, not silently rewritten. No data is
automatically dropped or balanced using the generated labels. They must never
be added to current-time forecasting inputs because both future image and future
report were used to create them. Human validation remains needed before claiming
clinical annotation accuracy.

## Runtime

Local model snapshot: `Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
vLLM 0.22.1, BF16, TP=1 replicas, 8,192 context tokens, up to 16 sequences per GPU.
Each image is aspect-preserving resized by the model processor to at most
1,048,576 pixels (minimum processor area 262,144); this is a first screening pass,
not full-resolution adjudication. Original images remain accessible in the review
gallery. Reports are not token-truncated. Overlength/decoding failures are explicit
request failures, not semantic classes. Thinking is disabled; temperature 0, seed
20260909, JSON schema constrained output, 512 output-token cap with incomplete
generations rejected.

If an output hits its length limit, only that failed pair is retried with a
1,024/2,048-token budget (bounded by remaining context) and schema-enforced
1–240-character evidence fields. A real repeated-text failure was reproduced and
resolved with this grammar constraint. Successful retries store their generated
class fields without edits, record `generation_max_tokens`, and receive `bounded_evidence_retry`
so they always enter manual review routing. Failed response bodies are retained.
This is not post-hoc truncation of a report or fabrication of missing JSON fields.
The initial successful records remain intact; `config_before_length_retry.json`
and `source_launch1/` preserve the initial run configuration and code.

The summary reports both raw model classes and conservative review routing.
Disagreement between image/report/overall labels, uncertainty, technical-confound
flags, low model confidence, or schema recovery sends a pair to `needs_review`.
That does not mean the pair is unusable. No records are deleted by review routing.

The host has an unavailable physical GPU 4. `vlm_runtime/sitecustomize.py` applies
two opt-in, process-local workarounds: skip only vLLM's advisory all-device name
scan, and resolve UUID device selections correctly for NVML. It does not suppress
memory checks or device execution errors. GPU selection is by UUID because CUDA
renumbers ordinals after excluding the failed device. No shared library is edited.
The installed FlashInfer sampling JIT is incompatible with local CUB headers;
`VLLM_USE_FLASHINFER_SAMPLER=0` selects the supported native PyTorch sampler.
For the large vocabulary, `MIMIC_VLLM_PYTORCH_TOPK=1` additionally selects vLLM's
existing PyTorch top-k/top-p mask function for warmup instead of compiling the
Triton sort. Actual screening uses temperature-zero greedy decoding. An opt-in
SIGUSR1 stack-dump handler is available for diagnosing owned server processes.

## Reproduce

Use `/home/data2/chk/workspace/2026/.venv/bin/python` as `python` below. Run from
this folder. All results are under `runs/full_20260909/vlm_qwen35_9b/`.

```bash
python audit_pair_funnel.py --run runs/full_20260909
python prepare_vlm.py --run runs/full_20260909 --out runs/full_20260909/vlm_qwen35_9b

# Repeat with distinct free GPUs and ports as needed. The launcher checks that
# the selected GPU is free, listens only on localhost, and records its PID.
python serve_vlm.py --gpu 2 --port 8120 --out runs/full_20260909/vlm_qwen35_9b

# Smoke test after /health is ready; train-only and separate from the full job.
python screen_vlm.py --manifest runs/full_20260909/vlm_qwen35_9b/manifest_expanded.jsonl \
  --out runs/full_20260909/vlm_qwen35_9b/smoke_train32 \
  --scope expanded --train-only --limit 32 --concurrency 8

# Full job; supply all healthy localhost replicas. No --limit means all pairs.
python run_vlm_job.py --manifest runs/full_20260909/vlm_qwen35_9b/manifest_expanded.jsonl \
  --out runs/full_20260909/vlm_qwen35_9b/full \
  --server-state runs/full_20260909/vlm_qwen35_9b \
  --scope expanded --concurrency 16 --endpoints http://127.0.0.1:8120 \
  --stop-owned-servers
```

The coordinator refreshes descriptive counts every five minutes, retries missing
pairs, writes `completion.json` only on exact coverage, and optionally releases
only its recorded servers after completion (PID start-time and command checked).
For long jobs launch it with a detached process/session; closing the chat must not
interrupt the sweep. Resume with exactly the same protocol/model/manifest/scope.
Do not run a second writer into the same output database; file locks prevent it.

`full/progress.json`: live completion, class counts among completed pairs and ETA.
`full/results.sqlite`: one row per successful pair, raw API response, token usage,
timestamps and endpoint; failed attempts are separate, append-only rows.
`full/summary/statistics.md`, `.json`, `.csv`: coverage and counts/percentages by
original subset, cohort tiers, split, gap, view matching and admission linkage.
`full/summary/pair_labels.jsonl`: reusable pair-ID labels and review flags.
`full/summary/review.html`: original image/report and weak-label examples.
`full/completion.json`: final exact-coverage marker; absent means not complete.

Partial-result percentages use completed pairs as denominator, explicitly show
coverage, and are never labeled final whole-cohort prevalence. Tier groups overlap
and must not be added. Statistics are descriptive: no causal interpretation or
unjustified significance testing. Tests exercise two-image requests, full-report
preservation, resume uniqueness, partial denominator handling and schema failures.
