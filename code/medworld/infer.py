"""Encode/predict a state or decode an exported state with its matching checkpoint."""
import argparse
from pathlib import Path

from .gpu import acquire_gpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--image")
    inputs.add_argument("--state", help="Standalone predicted_state.pt artifact")
    parser.add_argument("--report-file", help="Optional report of the input observation")
    parser.add_argument("--delta-hours", type=float, help="Positive forecast; negative retrodiction; omit for current state")
    parser.add_argument("--save-state")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpu", default="auto")
    args = parser.parse_args()
    if args.state and (args.report_file or args.delta_hours is not None):
        parser.error("Standalone decoding accepts only the state and matching checkpoint")
    for path in (args.out, args.save_state):
        if path and Path(path).exists():
            parser.error(f"Output exists: {path}")
    lock, device = acquire_gpu(args.gpu)
    import torch
    from PIL import Image, ImageOps
    from .datasets.protocol import _sha256
    from .runtime import atomic_json, load_model
    model, checkpoint = load_model(args.checkpoint, device)
    identity = _sha256(Path(args.checkpoint))
    with torch.no_grad():
        if args.state:
            artifact = torch.load(args.state, weights_only=True, map_location="cpu")
            if artifact.get("format_version") != 1 or artifact.get("checkpoint_sha256") != identity:
                raise ValueError("State artifact belongs to a different checkpoint or format")
            state = artifact["state"]
        else:
            with Image.open(args.image) as image:
                image = ImageOps.pad(image.convert("RGB"), (512, 512),
                                     method=Image.Resampling.BICUBIC, color="black")
            text = Path(args.report_file).read_text() if args.report_file else ""
            state = model.encode([image], [text])
            if args.delta_hours is not None:
                state = model.world(state, torch.tensor([args.delta_hours], device=device))
            artifact = {"format_version": 1, "state": state.cpu(), "checkpoint_sha256": identity,
                        "delta_hours": args.delta_hours}
        reports = model.decode_state(state, args.max_new_tokens)
    if args.save_state:
        Path(args.save_state).parent.mkdir(parents=True, exist_ok=True)
        torch.save(artifact, args.save_state)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out, {"reports": reports, "state_shape": list(state.shape),
                           "checkpoint_sha256": identity, "delta_hours": artifact.get("delta_hours")
                           if not torch.is_tensor(artifact.get("delta_hours")) else artifact["delta_hours"].tolist()})
    if lock is not None:
        lock.close()


if __name__ == "__main__":
    main()
