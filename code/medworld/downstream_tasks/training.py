"""Current-task objectives, combined with temporal prediction in joint training."""
from .classification import finding_loss
from .common.spatial import spatial_loss

def current_loss(model, task, batch, *, return_state=False):
    state = model.encode(batch["images"], spatial=task in ("segmentation", "sr"))
    if task == "report":
        loss = model.report.loss(state, batch["report_targets"])
    elif task == "classification":
        loss = finding_loss(model.classification(state), batch["labels"], batch["label_mask"],
                            model.pos_weight)
    elif task in ("segmentation", "sr"):
        pixels, mask = batch["pixels"].to(model.device), batch["mask"].to(model.device)
        prediction = getattr(model, task)(pixels, state)
        loss = spatial_loss(task, prediction, batch["targets"].to(model.device), mask)
    else:
        raise ValueError(f"Unsupported task {task}")
    parts = {task: loss.detach()}
    return (loss, parts, state) if return_state else (loss, parts)
