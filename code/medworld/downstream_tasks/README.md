# Downstream tasks

All tasks share `common/decoder.py`: Qwen image tokens are projected into a Transformer, optionally conditioned on all eight slots. Image tokens remain available in both experimental arms.

| Task | Additional input | Output head | Loss / metrics |
|---|---|---|---|
| classification | none | Mean pooling + linear | Masked BCE; AUROC/AP |
| segmentation | none | Token grid, upsampling, convolution | BCE + Dice; Dice |
| vqa | question | Shared decoded image tokens + question → language decoder | Answer-token CE; label-set EM/micro-F1 |

`registry.py` is the task list; `training.py` dispatches task losses. VQA text generation lives in `text/`. Slots are constructed without questions or current reports. Temporal latent prediction remains in the model as auxiliary representation learning, with no downstream temporal/report task.
