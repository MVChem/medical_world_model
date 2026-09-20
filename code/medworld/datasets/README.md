# Data

`current.py` provides source-image classification and segmentation; `vqa.py` adds official MIMIC-CXR-VQA images/questions/answers. Questions and images are inputs; answers are supervision only. No current report is returned to any downstream task.

`unified.py` enforces global patient holdouts across all three tasks and the temporal representation-learning data. Test/human-test take precedence over validation, which takes precedence over training. Rows are removed from conflicting lower-priority splits, never moved into held-out sets. Counts and source hashes are saved in each run's `data_protocol.json`.

`temporal.py` supplies source/target observations and signed intervals for auxiliary latent prediction. It is not a downstream classification or report task. `pixels.py` reconstructs source canvases/masks on demand; only original CXAS supervised labels may be read as arrays. No pixel/feature caches are used.
