"""Continuously updated numerical report and unretouched attention figures."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from medworld.runtime import atomic_json
from . import CONCEPTS


def attention_panel(image, maps, labels, title, path, spatial_normalization, valid=None):
    columns = 4 if len(maps) == 8 else 2
    rows = (len(maps) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(3.0 * columns, 3.0 * rows), squeeze=False)
    count = maps.shape[-1] * maps.shape[-2]
    if spatial_normalization and valid is not None:
        count = int((F.interpolate(torch.from_numpy(valid)[None, None].float(), maps.shape[-2:], mode="area") > .5).sum())
    shown = maps * count if spatial_normalization else maps
    vmax = max(float(shown.max()), 1e-6)
    for ax, values, label in zip(axes.flat, shown, labels):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1, extent=(0, 1, 1, 0))
        artist = ax.imshow(values, cmap="inferno", alpha=.6, vmin=0, vmax=vmax,
                           extent=(0, 1, 1, 0), interpolation="bilinear")
        ax.set_title(label, fontsize=10)
        ax.axis("off")
    for ax in list(axes.flat)[len(maps):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.colorbar(artist, ax=axes.ravel().tolist(), shrink=.65,
                 label=("weight / uniform valid-pixel weight" if valid is not None else "weight / uniform spatial weight")
                 if spatial_normalization else "attention weight")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def render_run(out, split="validate"):
    directory = out / split
    protocol = json.loads((out / "protocol.json").read_text())
    depths = protocol["cache_protocols"]["segmentation"].get("vision_depths_1based")
    labels = [f"S{i} / layer {j}" for i, j in zip(range(5, 9), depths)] if depths else [f"S{i}" for i in range(5, 9)]
    for path in directory.glob("*_attention.npz"):
        arrays = np.load(path)
        task = path.stem.removesuffix("_attention")
        source = arrays["input"][0, 0]
        attention_panel(source, arrays["encoder_attention"][0], labels,
            f"{task}: actual encoder slot-to-patch attention", directory / f"{task}_encoder.png", True)
        attention_panel(source, arrays["decoder_attention64"][0], [f"S{i}" for i in range(1, 9)],
            f"{task}: actual spatial-query to slot attention", directory / f"{task}_decoder.png", False)
        attention_panel(source, arrays["semantic_attention"][0], list(CONCEPTS),
            f"{task}: actual phrase-query attention", directory / f"{task}_semantic.png", True, arrays["valid"][0, 0])


def comparison(run, results):
    selected = [r for r in results if r["seed"] == 42 and r["variant"] in
                ("image_only", "slots", "featup", "featup_semantic", "image_only_featup")]
    order = {v: i for i, v in enumerate(("image_only", "slots", "featup", "featup_semantic", "image_only_featup"))}
    selected.sort(key=lambda r: order[r["variant"]])
    for task in ("segmentation", "sr"):
        entries = []
        for row in selected:
            path = Path(row["path"]) / "validate" / f"{task}_attention.npz"
            if path.exists():
                entries.append((row, np.load(path)))
        if len(entries) < 2:
            continue
        fig, axes = plt.subplots(2, len(entries) + 2, figsize=(3 * (len(entries) + 2), 6))
        sample = entries[0][1]
        source, target, valid = sample["input"][0, 0], sample["reference"][0], sample["valid"][0, 0]
        for ax in axes.flat:
            ax.axis("off")
        axes[0, 0].imshow(source, cmap="gray", vmin=0, vmax=1)
        axes[0, 0].set_title("Input / LR" if task == "sr" else "Input")
        axes[0, -1].imshow(target[0] if task == "sr" else target.argmax(0), cmap="gray" if task == "sr" else "viridis")
        axes[0, -1].set_title("Reference")
        errors = []
        for row, data in entries:
            prediction = data["prediction"][0]
            if task == "segmentation":
                prediction = 1 / (1 + np.exp(-np.clip(prediction, -50, 50)))
                error = np.abs(prediction[:len(target)] - target).mean(0) * valid
            else:
                prediction = prediction.clip(0, 1)
                error = np.abs(prediction[0] - target[0]) * valid
            errors.append(error)
        vmax = max(max(float(e.max()) for e in errors), 1e-5)
        for col, ((row, data), error) in enumerate(zip(entries, errors), 1):
            prediction = data["prediction"][0]
            if task == "segmentation":
                axes[0, col].imshow(source, cmap="gray", vmin=0, vmax=1, extent=(0, 256, 256, 0))
                for organ, color in zip(range(len(target)), ("cyan", "yellow", "magenta")):
                    mask = prediction[organ] > 0
                    if mask.any() and not mask.all():
                        axes[0, col].contour(mask, levels=[.5], colors=[color], linewidths=.7)
            else:
                axes[0, col].imshow(prediction[0].clip(0, 1), cmap="gray", vmin=0, vmax=1)
            axes[0, col].set_title(row["variant"], fontsize=9)
            artist = axes[1, col].imshow(error, cmap="magma", vmin=0, vmax=vmax)
            axes[1, col].set_title("Absolute error / common scale", fontsize=8)
        fig.colorbar(artist, ax=axes[1].tolist(), shrink=.7)
        fig.suptitle(f"{task}: fixed validation case; no outcome-based case selection")
        fig.savefig(run / f"{task}_comparison.png", dpi=140, bbox_inches="tight")
        fig.savefig(run / f"{task}_comparison.pdf", bbox_inches="tight")
        plt.close(fig)


def build(run, render=False):
    run = Path(run)
    results = []
    for path in sorted((run / "jobs").glob("*/summary.json")):
        row = json.loads(path.read_text())
        row["path"] = str(path.parent.resolve())
        results.append(row)
    atomic_json(run / "aggregate.json", results)
    lines = ["# Sparse spatial alignment pilot", "", "Fixed protocol; all scores below are raw units (Dice/SSIM 0–1, PSNR dB).",
             "Incomplete budgets are marked; compare the same seed and completed update count.", "",
             "| Variant | Seed | Updates | Budget complete | Test Dice | Human Dice | Test PSNR | Test SSIM |",
             "|---|---:|---:|---|---:|---:|---:|---:|"]
    def value(row, split, task, metric):
        result = row["evaluations"].get(split, {}).get(task, {}).get(metric)
        return f"{result:.5f}" if result is not None else "pending"
    for row in results:
        lines.append(f"| {row['variant']} | {row['seed']} | {row['steps']} | {row['complete_budget']} | " +
            " | ".join((value(row, "test", "segmentation", "dice"), value(row, "human_test", "segmentation", "dice"),
                       value(row, "test", "sr", "psnr"), value(row, "test", "sr", "ssim"))) + " |")
    if not results:
        lines += ["", "No completed training/evaluation results are available."]
    state_path = run / "status.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        lines += ["", f"Queue state: `{state['state']}`. GPU indices: 0, 1, 2, 6, 7.", "",
                  "| Job | State | GPU | Return code |", "|---|---|---|---|"]
        lines += [f"| {j['id']} | {j['state']} | {j.get('gpu', '')} | {j.get('returncode', '')} |" for j in state["jobs"]]
        if state.get("reason"):
            lines += ["", state["reason"]]
    # Paired seed / update-count contrasts; no significance claim from a pilot.
    lines += ["", "## Matched contrasts", "", "Positive deltas favor the first condition; compare only equal completed budgets."]
    for first, second in (("slots", "image_only"), ("slots", "visual_slots"), ("featup", "slots"),
                           ("featup_semantic", "featup"), ("featup_semantic", "image_only_featup")):
        for task, metric in (("segmentation", "dice"), ("sr", "psnr")):
            deltas = []
            for a in results:
                if a["variant"] != first or not a["complete_budget"]:
                    continue
                matches = [b for b in results if b["variant"] == second and b["seed"] == a["seed"]
                           and b["complete_budget"] and b["steps"] == a["steps"]]
                if matches:
                    av = a["evaluations"].get("test", {}).get(task, {}).get(metric)
                    bv = matches[0]["evaluations"].get("test", {}).get(task, {}).get(metric)
                    if av is not None and bv is not None:
                        deltas.append(av - bv)
            if deltas:
                lines.append(f"- {first} − {second}, test {metric}: mean delta {np.mean(deltas):+.5f}; "
                             f"{len(deltas)} matched seed(s), seed deltas {', '.join(f'{d:+.5f}' for d in deltas)}.")
    lines += ["", "## Scope", "",
        "This run freezes the checkpoint's VLM/JEPA backbones and image-only fusion states; visual slot queries/readouts and spatial heads train.",
        "Existing segmentation/SR task targets are unchanged. Additional alignment uses frozen visual features and native VLM soft crop scores, without new region labels.",
        "The method borrows FeatUp's multiview consistency objective; it is not a reproduction of FeatUp's learned JBU/downsampler.",
        "Test/human-test data never provide optimizer updates or checkpoint selection. Attention maps are not lesion ground truth.",
        "", "## Artifacts", "",
        "- `aggregate.json`: full numerical results and run paths.",
        "- `cache/{segmentation,sr}/semantic_diagnostics.json`: raw weak-teacher probabilities and coverage.",
        "- `jobs/*/{validate,test,human_test}/*_attention.npz`: raw predictions, encoder/decoder/semantic attention, targets and zero-slot sensitivity.",
        "- `jobs/*/gradient_audit.json`: task gradients reaching slot queries and attention modules.",
        "- `*_comparison.png` / `.pdf`: fixed validation case comparisons with shared error scale.", ""]
    (run / "REPORT.md").write_text("\n".join(lines))
    if render:
        for row in results:
            if row["seed"] == 42:
                render_run(Path(row["path"]))
        comparison(run, results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    build(args.run, args.render)
