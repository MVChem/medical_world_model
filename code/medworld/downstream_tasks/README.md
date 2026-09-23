# Downstream tasks

The `raw_input_v1` architecture takes raw inputs directly into its task
decoder: a learned image patch projection and optional UTF-8 report embeddings
feed a shared two-layer Transformer. The slots arm appends all eight slots as
extra conditions; original input tokens remain available in both arms. No Qwen
visual features enter this downstream path.

| Task | Additional input | Output head | Loss / metrics |
|---|---|---|---|
| classification | none | Mean pooling + linear | Masked BCE; AUROC/AP |
| segmentation | image required | Spatial token grid, upsampling, convolution | BCE + Dice; per-dataset mean IoU/Dice |
| vqa | question | Decoded input tokens + question → small byte autoregressive decoder | Answer-byte CE; label-set EM/micro-F1 |

The VQA decoder is trained from scratch with identical initialization in both
arms. It does not contain a pretrained Qwen language model. The raw task/VQA
decoders' context, answer and generation token budgets count UTF-8 bytes and
special tokens; the separate slots state encoder still tokenizes its context
with Qwen.

`registry.py` is the task list; `training.py` dispatches task losses.
`common/decoder.py` implements `TaskDecoder`; `text/decoder.py` implements
`TextDecoder` for VQA. The input API supports images, reports, or both;
the current dataset adapter supplies images and VQA questions, never the reports
used to derive target labels. Segmentation requires an image grid. The slots
branch requires images and rejects report-only inputs because its encoder
constructs visual slots. Questions do not enter slot construction.

Temporal prediction, EMA and optional VSSC are representation-learning objectives
used only by the slots branch. The raw-input baseline constructs none of those
components and uses supervised task losses only.
