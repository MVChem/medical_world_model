# Datasets

`UnifiedData` is the training data entry point: current tasks use `current.py`, and temporal prediction uses `temporal.py`.
`protocol.py` checks manifest hashes and patient splits; `pixels.py` decodes source images and masks and generates LR inputs online.
Patients cannot overlap across training, validation, and test splits. Conflicting samples are removed from lower-priority splits rather than moved into test sets.

## Configurable paths

Override these keys in the training JSON configuration. Relative paths are resolved against the repository root; absolute paths are also supported.
Defaults are centralized in `config.py` and retain existing artifact locations to preserve the data protocol. Loaders do not import code from those locations.

| Key | Required contents |
|---|---|
| current_data | observations.jsonl, manifest.json, segmentation.json, seg_probs.npy |
| dense_data | observations.jsonl and manifest.json; records point to source chest images and Montgomery masks |
| selection_file | JSON file containing patient split overrides |
| classification_data | classification_data.json and classification_observations.jsonl |
| baseline_data | protocol.json and fixed validation/test input and reference manifests under cohort/ |
| temporal_data | observations.jsonl, train.jsonl, validate.jsonl, test.jsonl |
| qwen / jepa | Local Qwen weights directory / V-JEPA checkpoint |
| clinical_weights | bert-base-uncased directory and chexbert.pth |

The `image` and `masks` paths in manifests must remain valid. Moving source data requires updating manifests and their protocol hashes.
The current adapter preserves historical sample selection and count checks; it does not build new cohorts by scanning all MIMIC data.

MIMIC-CXR provides images, reports, and labels. MIMIC-IV contributes the previously established admission linkage.
Temporal inputs consist of the source image, source report, and signed actual time interval. `ehr_text` is not used as model input.
Temporal evaluation performs source-only inference before accessing target references to prevent target leakage.
Human segmentation masks come from Montgomery; fixed CXAS pseudo-labels are used only as segmentation supervision.

Each image is read from its source file and reconstructed on a 512×512 canvas. Super-resolution generates 128×128 LR inputs online and uses 512×512 HR targets.
Prefetching is in memory only. Disk image and feature caches are neither read nor written.
