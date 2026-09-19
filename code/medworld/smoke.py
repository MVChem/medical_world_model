"""Real-model acceptance checks; these do not estimate clinical quality."""
import gc

import torch

from .downstream_tasks.data import _sha256


def audit_model(model, data, out):
    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)
    model.eval()
    batch = data.batch("temporal", "validate", [0])
    _, _, graph = model.temporal_loss(batch, audit=True)
    parameters = [model.encoder.slot_queries, model.world.output.weight, model.report.projection[1].weight]
    gradients = torch.autograd.grad(graph["report"], parameters)
    norms = {name: float(gradient.float().norm()) for name, gradient in
             zip(("online_slot_queries", "world_output", "report_projection"), gradients)}
    if any(not value > 0 for value in norms.values()) or graph["target"].requires_grad:
        raise AssertionError("Report gradient chain or target stop-gradient failed")
    online = dict(model.encoder.named_parameters())
    teacher = dict(model.target.encoder.named_parameters())
    frozen = [n for n, p in online.items() if not p.requires_grad]
    if not all(online[n] is teacher[n] for n in frozen):
        raise AssertionError("Frozen target base parameters were duplicated")
    if any(online[n] is teacher[n] for n in model.target.parameter_names):
        raise AssertionError("EMA adapters unexpectedly alias online parameters")
    if any(p.grad is not None for p in model.target.parameters()):
        raise AssertionError("Target accumulated gradients")
    result = {"report_only_gradient_norms": norms, "target_requires_grad": False,
              "frozen_encoder_target_parameters_shared": len(frozen),
              "independent_ema_parameters": len(model.target.parameter_names),
              "ema_updates": int(model.target.updates),
              "ema_online_slot_difference": float((online["slot_queries"] - teacher["slot_queries"]).detach().norm())}
    del graph, gradients
    model.zero_grad(set_to_none=True)
    gc.collect()
    checkpoint = out / "stage2.pt"
    checkpoint_hash = _sha256(checkpoint)
    with torch.no_grad():
        evidence = data.batch("temporal", "validate", [0, 1], source_only=True)
        state = model.predict_state(**evidence)
        artifact = {"state": state.cpu(), "checkpoint_sha256": checkpoint_hash,
                    "delta_hours": evidence["delta_hours"], "format_version": 1}
        torch.save(artifact, out / "predicted_state.pt")
        text = model.decode_state(state)
        loaded = torch.load(out / "predicted_state.pt", weights_only=True, map_location="cpu")
        if model.decode_state(loaded["state"]) != text:
            raise AssertionError("Standalone CPU-state decoding differs")
        result["reports"] = text
        result["signed_delta_hours"] = evidence["delta_hours"].tolist()
        result["cpu_state_roundtrip_equal"] = True
        target_reports = data.batch("temporal", "validate", [0, 1])["report_targets"]
        result["state_condition_ce"] = {
            "predicted": float(model.report.loss(state, target_reports)),
            "swapped": float(model.report.loss(state.flip(0), target_reports)),
            "zero": float(model.report.loss(torch.zeros_like(state), target_reports)),
        }
    # Reconstruct the decoder/model from disk: no access to the original
    # observations is needed to decode the exported latent tensor.
    from .runtime import load_model
    restored, _ = load_model(checkpoint, str(model.device))
    with torch.no_grad():
        if restored.decode_state(loaded["state"]) != text:
            raise AssertionError("Checkpoint reload changed standalone decoding")
        restored_state = restored.predict_state(**evidence)
        max_error = float((restored_state - state).abs().max())
        if max_error > 1e-5:
            raise AssertionError(f"Checkpoint reload changed state: {max_error}")
    if hasattr(model, "featup_teacher"):
        result["featup"] = {}
        for task in ("segmentation", "sr"):
            batch = data.batch(task, "validate", [0])
            loss, parts = model.current_loss(task, batch)
            head = getattr(model, task)
            vision_lora = next(p for n, p in model.encoder.vision.named_parameters() if "lora_b" in n)
            parameters = [model.encoder.slot_queries, model.encoder.visual_readouts[0][1].weight,
                          vision_lora, head.upsample.kernel[-1].weight, head.feature.weight]
            names = ["slot_queries", "visual_readout", "vision_lora", "guided_kernel", "feature_projection"]
            gradients = torch.autograd.grad(loss, parameters)
            norms = {name: float(g.float().norm()) for name, g in zip(names, gradients)}
            if any(not value > 0 for value in norms.values()):
                raise AssertionError(f"Missing FeatUp gradient: {task}: {norms}")
            with torch.no_grad():
                pixels = batch["pixels"].to(model.device)
                expected = head(pixels, model.encode(batch["images"], spatial=True))
                recovered = getattr(restored, task)(pixels, restored.encode(batch["images"], spatial=True))
                torch.testing.assert_close(recovered, expected, rtol=0, atol=1e-5)
                _, reloaded_parts = restored.current_loss(task, batch)
                torch.testing.assert_close(parts[task + "_feature"], reloaded_parts[task + "_feature"],
                                           rtol=0, atol=1e-5)
            result["featup"][task] = {"gradient_norms": norms,
                "feature_loss": float(parts[task + "_feature"]),
                "prediction_reload_max_error": float((expected - recovered).abs().max())}
            del gradients, loss
        if any(p.requires_grad or p.grad is not None for p in model.featup_teacher.parameters()):
            raise AssertionError("Feature teacher must stay frozen")
        result["featup"]["teacher_frozen"] = True
    result.update(checkpoint_roundtrip_equal=True, state_reload_max_error=max_error,
                  audit_peak_gpu_memory_gib=(torch.cuda.max_memory_allocated(model.device) / 1024**3
                                            if model.device.type == "cuda" else None),
                  gpu_memory_scope="Audit includes original and reloaded models; training step peaks are in metrics.jsonl",
                  quality_evaluation="Interface/gradient smoke only; not a report-quality benchmark")
    del restored
    return result
