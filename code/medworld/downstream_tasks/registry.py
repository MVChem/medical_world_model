"""Current task order, label vocabulary and explicitly pending task adapters."""
TASKS = ("classification", "report", "segmentation", "sr")
SPLITS = ("train", "validate", "test", "human_test")
FINDINGS = (
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
    "Pleural Effusion", "Pleural Other", "Pneumonia", "Pneumothorax",
    "Support Devices",
)
PENDING_TASKS = {
    "vqa": "Official MIMIC-CXR-VQA data and full benchmark adapter are pending; derived QA is excluded.",
    "grounding": "MS-CXR lesion-phrase data and adapter are pending; anatomy boxes are excluded.",
}
