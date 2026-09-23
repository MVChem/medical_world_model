"""Raw inputs are always available; slots add an optional conditioning branch."""

RAW_INPUT = "raw_input_v1"


def architecture(cfg):
    value = cfg.get("architecture", RAW_INPUT)
    if value != RAW_INPUT:
        raise ValueError(f"Unsupported MedWorld architecture: {value!r}; expected {RAW_INPUT!r}")
    return value


def uses_slot_branch(cfg):
    return cfg["slot_conditioning"]


def uses_temporal(cfg):
    return cfg["slot_conditioning"] and cfg["latent_weight"] > 0


def normalize_baseline(cfg):
    """Remove auxiliary objectives from a raw-input task-only training arm."""
    if not cfg["slot_conditioning"]:
        cfg["latent_weight"] = 0.0
        cfg["visual_consistency_weight"] = 0.0
    return cfg


def baseline_label(cfg):
    architecture(cfg)
    return "Raw-input task-only baseline"


def require_reviewed_data(cfg, data):
    architecture(cfg)
    if getattr(data.current, "metadata", {}).get("schema") != "medworld-prepared-v2":
        raise ValueError("Training requires human-reviewed medworld-prepared-v2 data")
