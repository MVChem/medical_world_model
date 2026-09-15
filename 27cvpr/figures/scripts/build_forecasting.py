#!/usr/bin/env python3
"""Render existing 0.8B forecasting outputs without rerunning a model.

Selection uses reference labels and explicit report-text consistency rules only.
This figure is a historical pilot review artifact, not a 9B or full 4+4 result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "code/medworld_table1/data/linked_20260913_16k"
RUN = ROOT / "code/medworld_table1/runs/overnight_20260913"
FINDINGS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion", "Pneumothorax"]
FOCUS = 4
COLORS = dict(ink="#172A3A", muted="#627184", line="#D7E1E9", full="#4979A4", slots="#177F7D", ref="#816642")


def read_jsonl(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def effusion_sentences(report):
    # Preserve exact source wording, stripping only section headings.
    return [re.sub(r"^(FINDINGS|IMPRESSION):\s*", "", sentence.strip())
            for sentence in re.split(r"(?<=[.!?])\s+|\n", report)
            if re.search("effusion", sentence, re.I)]


def explicit_absence(report):
    for sentence in effusion_sentences(report):
        absent = re.search(
            r"\bno (?:evidence of )?(?:(?:appreciable|complicating) )?pleural effusions?\b"
            r"|\bno pneumothorax or pleural effusion\b", sentence, re.I)
        ambiguous = re.search(r"\b(likely|possible|left|right|larger)\b", sentence, re.I)
        if absent and not ambiguous:
            return sentence
    return None


def explicit_presence(report):
    for sentence in effusion_sentences(report):
        if not re.search(r"\b(no|not|without|resolution|resolved|likely|could|suggesting|possible|may|question)\b", sentence, re.I):
            return sentence
    return None


def report_excerpt(report):
    sentences = effusion_sentences(report)
    return sentences[0] if sentences else "[No effusion-specific sentence in generated report.]"


def choose_cases(rows, observations, labels):
    selected, exclusions, used = [], [], set()
    for kind, source_label, target_label in [("Label onset", 0, 1), ("Label resolution", 1, 0), ("Label persistence", 1, 1)]:
        for i in sorted(range(len(rows)), key=lambda index: rows[index]["id"]):
            row = rows[i]
            if row["patient"] in used or (labels["current"][i][FOCUS], labels["target"][i][FOCUS]) != (source_label, target_label):
                continue
            source, target = observations[row["source"]], observations[row["target"]]
            source_quote = (explicit_presence if source_label else explicit_absence)(source["report"])
            target_quote = (explicit_presence if target_label else explicit_absence)(target["report"])
            if not source_quote or not target_quote:
                exclusions.append(dict(kind=kind, pair_id=row["id"], reason="Reference report failed explicit mention/absence consistency screen", source_pass=bool(source_quote), target_pass=bool(target_quote)))
                continue
            if not Path(source["image"]).is_file() or not Path(target["image"]).is_file():
                raise FileNotFoundError("Selected source or follow-up image missing")
            selected.append(dict(index=i, kind=kind, row=row, source=source, target=target,
                                 source_quote=source_quote, target_quote=target_quote,
                                 source_label=source_label, target_label=target_label))
            used.add(row["patient"])
            break
        else:
            raise ValueError(f"No eligible case for {kind}")
    return selected, exclusions


def wrapped(text, width):
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def compact_source_quote(quote):
    # Keep the effusion-bearing end of a long sentence, with explicit omission.
    if len(quote) <= 130:
        return quote
    tail = quote[-125:]
    return "[…] " + tail.split(" ", 1)[1]


def label_name(value):
    return {0: "absent", 1: "present", -1: "uncertain", -2: "not mentioned"}[value]


def panel(fig, x, y, width, height, color="#FFFFFF"):
    ax = fig.add_axes([x, y, width, height])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((.006, .006), .988, .988, boxstyle="round,pad=0.002,rounding_size=0.024", facecolor=color, edgecolor=COLORS["line"], linewidth=.8))
    return ax


def text(ax, x, y, value, size=10, **kwargs):
    ax.text(x, y, value, fontsize=size, color=kwargs.pop("color", COLORS["ink"]), va="top", **kwargs)


def cxr(fig, x, y, width, height, path, caption, quote=None):
    ax = panel(fig, x, y, width, height, "#F8FAFC")
    image_ax = fig.add_axes([x + width * .035, y + height * .235, width * .93, height * .74])
    image_ax.set_facecolor("black")
    image_ax.imshow(Image.open(path).convert("L"), cmap="gray", vmin=0, vmax=255)
    image_ax.axis("off")
    text(ax, .05, .205, caption, 9.6, fontweight="bold")
    if quote:
        text(ax, .05, .131, wrapped(quote, 35), 8.4, color=COLORS["muted"])
    return ax


def model_card(fig, x, y, width, height, prediction, predicted_label, accent):
    ax = panel(fig, x, y, width, height)
    probability = prediction["scores"][FOCUS]
    text(ax, .075, .92, "Future effusion probability", 9.2, color=COLORS["muted"])
    text(ax, .075, .80, f"{100 * probability:.1f}%", 25, color=accent, fontweight="bold")
    ax.add_patch(Rectangle((.075, .55), .85, .027, facecolor="#E9EEF3", edgecolor="none"))
    ax.add_patch(Rectangle((.075, .55), .85 * probability, .027, facecolor=accent, edgecolor="none"))
    text(ax, .075, .485, "Generated report excerpt", 9.2, color=COLORS["muted"])
    excerpt = report_excerpt(prediction["report"])
    text(ax, .075, .395, wrapped(excerpt, 40), 10.1, linespacing=1.35)
    text(ax, .075, .09, f"Report label: {label_name(predicted_label)}", 8.8, color=COLORS["muted"])
    return excerpt


def render(cases, predictions, labels, out):
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(16, 11.0), facecolor="white")
    fig.text(.034, .974, "Longitudinal forecasting from the current clinical state", fontsize=20, fontweight="bold", color=COLORS["ink"], va="top")
    fig.text(.034, .938, "Historical Qwen3.5-0.8B pilot  |  matched full-token and eight learned-query-slot forecasters", fontsize=11.7, color=COLORS["muted"], va="top")
    xs = [.034, .198, .373, .591, .812]
    widths = [.149, .160, .203, .203, .154]
    headers = [("Current CXR", "available at prediction"), ("Copy Current", "copies current report"), ("Full-token baseline", "0.8B matched forecast model"), ("Eight-slot pilot", "0.8B; final-language-layer slots"), ("Observed follow-up", "REFERENCE ONLY")]
    for x, width, (title, subtitle) in zip(xs, widths, headers):
        fig.text(x + width / 2, .892, title, ha="center", va="top", fontsize=12.3, fontweight="bold", color=COLORS["ink"])
        fig.text(x + width / 2, .870, subtitle, ha="center", va="top", fontsize=8.7, color=COLORS["muted"])
    render_records = []
    for row_number, case in enumerate(cases):
        y = .592 - row_number * .25
        height = .214
        pair = case["row"]
        horizon = {0: "6–24 h", 1: ">24–72 h", 2: ">72–168 h", 3: ">168–720 h"}[pair["horizon"]]
        title = f"{chr(97 + row_number)})  {case['kind']}  {case['source_label']} → {case['target_label']}    |    Pleural effusion"
        fig.text(xs[0], y + height + .021, title, fontsize=11.4, color=COLORS["ink"], fontweight="bold", va="top")
        fig.text(.966, y + height + .020, f"Input horizon: {horizon}    ·    Observed gap: {pair['realized_gap_hours']:.1f} h", fontsize=9.5, color=COLORS["muted"], va="top", ha="right")
        cxr(fig, xs[0], y, widths[0], height, case["source"]["image"], f"Current label: {label_name(case['source_label'])}", f"{pair['view']} view · current image + report\n+ pre-image clinical history")
        copy = panel(fig, xs[1], y, widths[1], height, "#F8FAFC")
        text(copy, .07, .92, "Future effusion label", 9.2, color=COLORS["muted"])
        text(copy, .07, .79, label_name(case["source_label"]).capitalize(), 19.5, fontweight="bold")
        text(copy, .07, .64, "No continuous probability", 8.5, color=COLORS["muted"])
        text(copy, .07, .48, "Current report excerpt", 9.2, color=COLORS["muted"])
        displayed_copy_quote = compact_source_quote(case["source_quote"])
        text(copy, .07, .385, wrapped(displayed_copy_quote, 34), 9.3, linespacing=1.30)
        excerpts = {}
        for key, column, accent in [("no_slots", 2, COLORS["full"]), ("slots", 3, COLORS["slots"])]:
            i = case["index"]
            excerpts[key] = model_card(fig, xs[column], y, widths[column], height, predictions[key][i], labels[key]["predicted"][i][FOCUS], accent)
        cxr(fig, xs[4], y, widths[4], height, case["target"]["image"], f"Reference label: {label_name(case['target_label'])}")
        # Exact target excerpt is preserved in provenance; the image gets only a
        # short label to avoid obscuring the observed follow-up.
        render_records.append(dict(pair_id=pair["id"], source_quote=displayed_copy_quote, target_quote=case["target_quote"], generated_excerpts=excerpts))
    fig.text(.034, .064, "Selection: fixed finding; first eligible pair by sorted ID for each reference-label transition; distinct patients; explicit report-text consistency screen.", fontsize=9.2, color=COLORS["muted"], va="top")
    fig.text(.034, .046, "CheXbert report labels are not clinician-adjudicated changes. Probabilities come from finding heads; report labels can disagree. Follow-up images are never model inputs.", fontsize=9.2, color=COLORS["muted"], va="top")
    fig.text(.034, .028, "Pilot review artifact: these outputs do not evaluate the pending 9B run or the proposed multi-depth fusion/vision 4+4 architecture.", fontsize=9.2, color=COLORS["muted"], va="top")
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(out / f"forecasting.{suffix}", dpi=240, facecolor="white")
    plt.close(fig)
    return render_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "27cvpr/figures/generated")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(DATA / "test.jsonl")
    needed = {row[key] for row in rows for key in ("source", "target")}
    observations = {row["id"]: row for row in read_jsonl(DATA / "observations.jsonl") if row["id"] in needed}
    labels, predictions, sources, generations = {}, {}, [], {}
    for condition in ("no_slots", "slots"):
        directory = RUN / condition / "evaluation_test"
        labels[condition] = json.loads((directory / "chexbert_labels.json").read_text())
        predictions[condition] = read_jsonl(directory / "predictions.jsonl")
        generations[condition] = json.loads((directory / "generation.json").read_text())
        assert [prediction["id"] for prediction in predictions[condition]] == [row["id"] for row in rows]
        assert generations[condition]["count"] == len(rows) == 297
        assert generations[condition]["target_inputs"] is False
        for name in ("chexbert_labels.json", "predictions.jsonl", "generation.json"):
            path = directory / name
            sources.append(dict(path=str(path.relative_to(ROOT)), sha256=sha256(path)))
    assert labels["slots"]["current"] == labels["no_slots"]["current"]
    assert labels["slots"]["target"] == labels["no_slots"]["target"]
    cases, exclusions = choose_cases(rows, observations, labels["slots"])
    render_records = render(cases, predictions, labels, args.out)
    for path in (DATA / "test.jsonl", DATA / "observations.jsonl", Path(__file__).resolve()):
        sources.append(dict(path=str(path.relative_to(ROOT)), sha256=sha256(path)))
    provenance = dict(
        title="Historical 0.8B longitudinal forecasting pilot",
        status="Review artifact; not a 9B result and not the full multi-depth 4+4 architecture",
        run="overnight_20260913", generation=generations, findings=FINDINGS,
        focused_finding=FINDINGS[FOCUS], cohort=dict(n_pairs=len(rows), n_patients=len({row["patient"] for row in rows}), split="test"),
        selection="Fixed Pleural Effusion finding; for (0,1), (1,0), (1,1), first sorted pair ID from a distinct patient with both reports passing the script's explicit presence/absence screening. Selection never accesses prediction correctness or scores.",
        exclusions=exclusions, clinical_reference="CheXbert-extracted report labels, not clinician adjudicated. Persistence concerns only the selected finding. No disease-direction or severity claim.",
        inputs="Source image, current report, clinical history available by source image time, and discrete horizon bin. Exact observed gap is displayed only for reference.",
        copy_current="Current report copied verbatim as baseline; displayed exact effusion-specific source sentence. No continuous probability is assigned.",
        followup="Observed follow-up CXR; reference only; no future-image generation and no target inputs during prediction.",
        score_note="Finding-head probabilities and CheXbert labels of generated text are distinct interfaces and may disagree.",
        attention_audit=dict(
            frozen_visual_slots="Four intermediate vision blocks are mean-pooled in medworld_dense_baselines/frozen_slots_extract.py; no learned slot-to-patch attention exists in this path.",
            downstream_readout="medworld_stage1/networks.py SlotQueries attends over aggregate slots, not spatial patches; medworld_stage1/slot44_networks.py SpatialRead attends spatial queries to slots, the reverse direction of slot-readout-to-patch attention.",
            pilot_encoder="medworld_common/qwen.py StateEncoder appends eight learned slot tokens to the visual/text sequence and returns last-layer slot hidden states. Current wrapper exposes no attention weights; these are not the proposed multi-depth 4+4 readout. No attention visualization was fabricated."),
        cases=[dict(pair=case["row"], kind=case["kind"], source_image=case["source"]["image"], target_image=case["target"]["image"],
                    source_image_sha256=sha256(Path(case["source"]["image"])), target_image_sha256=sha256(Path(case["target"]["image"])),
                    current_report=case["source"]["report"], observed_report=case["target"]["report"],
                    current_labels=labels["slots"]["current"][case["index"]], observed_labels=labels["slots"]["target"][case["index"]],
                    predictions={key: dict(**predictions[key][case["index"]], report_labels=labels[key]["predicted"][case["index"]]) for key in predictions},
                    displayed=record) for case, record in zip(cases, render_records)],
        sources=sources)
    (args.out / "forecasting_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"outputs": [str(args.out / f"forecasting.{suffix}") for suffix in ("pdf", "png", "svg")],
                      "selected": [{"pair_id": case["row"]["id"], "transition": case["kind"], "gap_h": case["row"]["realized_gap_hours"]} for case in cases],
                      "excluded_before_selection": len(exclusions)}, indent=2))


if __name__ == "__main__":
    main()
