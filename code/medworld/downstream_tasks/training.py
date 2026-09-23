"""All task heads retain their native input; slots are optional conditions."""
import json
from .classification import finding_loss
from .segmentation.loss import segmentation_loss


def current_loss(model, task, batch, *, return_state=False):
    prepared = {"prepared": batch["prepared"]} if "prepared" in batch else {}
    if "reports" in batch:
        prepared["reports"] = batch["reports"]
    if task == "segmentation" and not batch.get("images"):
        raise ValueError("Segmentation requires an input image with spatial coordinates")
    features, slots = model.task_inputs(batch.get("images"), **prepared)
    if task == "classification":
        loss = finding_loss(model.classification(features), batch["labels"], batch["label_mask"], model.pos_weight)
    elif task == "segmentation":
        prediction = model.segmentation(features)
        loss = segmentation_loss(prediction, batch["targets"].to(model.device), batch["mask"].to(model.device))
    elif task == "vqa":
        answers = [json.dumps(answer, ensure_ascii=False) for answer in batch["answers"]]
        loss = model.text.loss(features, batch["questions"], answers)
    else:
        raise ValueError(f"Unsupported task: {task}")
    parts = {task: loss.detach()}
    return (loss, parts, slots) if return_state else (loss, parts)
