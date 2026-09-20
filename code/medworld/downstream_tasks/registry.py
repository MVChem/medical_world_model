"""Current task order, label vocabulary and explicitly pending task adapters."""
TASKS = ("classification", "segmentation", "vqa")
SPLITS = ("train", "validate", "test", "human_test")
FINDINGS = (
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
    "Pleural Effusion", "Pleural Other", "Pneumonia", "Pneumothorax",
    "Support Devices",
)
PENDING_TASKS = {}
