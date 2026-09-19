# Downstream tasks

This directory contains task models, losses, and metrics. Data loading lives in `../datasets/`; evaluation execution and result aggregation live in `../evaluation/`.

| Package | Inputs | Decoder | Loss | Metrics |
|---|---|---|---|---|
| classification | 8 slots | ClassificationHead | Masked BCE | AUROC/AP |
| report | 8 slots | ReportDecoder | decoder.loss: teacher-forced CE | Text generation diagnostics; clinical scoring in evaluation/ |
| segmentation | Input image + 4 visual slots | SegmentationHead | BCE + Dice | Dice |
| super_resolution | LR image + 4 visual slots | SuperResolutionHead | Masked MSE | PSNR/SSIM |

Segmentation and super-resolution decoders inherit the standard architecture in `common/spatial.py`, with separate instances and independent parameters.
`common/` contains positional encodings, shape checks, and shared model components. It does not load data or aggregate experiments.
`registry.py` defines stable task identifiers (super-resolution remains `sr`). `training.py` dispatches current-task losses within joint current-task and temporal training.
`__init__.py` files expose public interfaces. To add a task, define its model, loss, and metrics in a separate package, then connect it to datasets, registry, training, and evaluation.
