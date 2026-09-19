"""Current-task objectives, used by Stage 1 and Stage 2 replay."""
from .classification import finding_loss
from .spatial import spatial_loss

def current_loss(model, task, batch):
    state = model.encode(batch["images"], spatial=task in ("segmentation", "sr"))
    if task == "report":
        loss = model.report.loss(state, batch["report_targets"])
    elif task == "classification":
        loss = finding_loss(model.classification(state), batch["labels"], batch["label_mask"],
                            model.pos_weight)
    elif task in ("segmentation", "sr"):
        pixels, mask = batch["pixels"].to(model.device), batch["mask"].to(model.device)
        weight = model.cfg.get("featup_feature_weight", 0.1)
        if task == "sr":
            weight *= model.cfg.get("featup_sr_scale", 0.001)
        if hasattr(model, "featup_teacher") and weight > 0:
            from .featup import consistency_loss
            prediction, field = getattr(model, task)(pixels, state, return_features=True)
            feature = consistency_loss(field, pixels, mask, batch["ids"],
                                       model.featup_teacher, model.processor, model.cfg)
            supervised = spatial_loss(task, prediction, batch["targets"].to(model.device), mask)
            loss = supervised + weight * feature
            return loss, {task: loss.detach(), task + "_supervised": supervised.detach(),
                          task + "_feature": feature.detach(),
                          task + "_feature_weighted": (weight * feature).detach()}
        prediction = getattr(model, task)(pixels, state)
        loss = spatial_loss(task, prediction, batch["targets"].to(model.device), mask)
    else:
        raise ValueError(f"Unsupported task {task}")
    return loss, {task: loss.detach()}
