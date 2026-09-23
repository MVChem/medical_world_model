"""Task order, classification vocabulary and reviewed segmentation channels."""
TASKS = ("classification", "segmentation", "vqa")
SPLITS = ("train", "validate", "test", "human_test")
FINDINGS = (
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
    "Pleural Effusion", "Pleural Other", "Pneumonia", "Pneumothorax",
    "Support Devices",
)

# Manual CXR organs and reviewed brain-tumor regions share a padded target
# tensor; per-example masks select only the channels annotated in that dataset.
MANUAL_SEGMENTATION_LAYOUT = ("lungs", "heart", "NETC", "SNFH", "ET", "RC")
