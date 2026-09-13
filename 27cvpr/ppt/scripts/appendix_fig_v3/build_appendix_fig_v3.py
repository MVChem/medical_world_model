from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image


BASE = Path(__file__).resolve().parents[2]
PAIR_SOURCE = BASE / "source" / "mimic_appendix_cases"
SERIAL_SOURCE = BASE / "prompt" / "mimic_longitudinal_appendix_image2" / "images"
DEFAULT_OUTPUT = BASE / "ppt" / "appendix_fig_v3.pdf"

PAGE_WIDTH = 7
PAGE_HEIGHT = 5

INK = "#111827"
SECONDARY = "#4B5563"
ACCENT = "#244C6F"
RULE = "#C7CDD4"


def rule(fig, x1, y1, x2, y2, width=0.65):
    fig.add_artist(
        Line2D(
            [x1, x2],
            [y1, y2],
            transform=fig.transFigure,
            color=RULE,
            linewidth=width,
            solid_capstyle="butt",
        )
    )


def arrow(fig, x1, x2, y, note=None):
    fig.add_artist(
        FancyArrowPatch(
            (x1, y),
            (x2, y),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.9,
            color=ACCENT,
        )
    )
    if note:
        fig.text(
            (x1 + x2) / 2,
            y + 0.018,
            note,
            ha="center",
            va="bottom",
            fontsize=5.4,
            linespacing=1.1,
            color=SECONDARY,
        )


def crop_to_panel(image_path: Path, panel_aspect: float, focus_y=0.43):
    image = np.asarray(Image.open(image_path).convert("L"))
    image_aspect = image.shape[1] / image.shape[0]

    if image_aspect > panel_aspect:
        crop_width = int(round(image.shape[0] * panel_aspect))
        left = max(0, (image.shape[1] - crop_width) // 2)
        return image[:, left : left + crop_width]

    crop_height = int(round(image.shape[1] / panel_aspect))
    center = int(round(image.shape[0] * focus_y))
    top = min(max(0, center - crop_height // 2), image.shape[0] - crop_height)
    return image[top : top + crop_height, :]


def radiograph(fig, image_path: Path, x, y, width, height):
    panel_aspect = (width * PAGE_WIDTH) / (height * PAGE_HEIGHT)
    image = crop_to_panel(image_path, panel_aspect)
    ax = fig.add_axes([x, y, width, height], zorder=2)
    ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="lanczos", aspect="auto")
    ax.set_axis_off()
    fig.add_artist(
        Rectangle(
            (x, y),
            width,
            height,
            transform=fig.transFigure,
            fill=False,
            edgecolor="#9CA3AF",
            linewidth=0.45,
            zorder=3,
        )
    )


def section_header(fig, x, y, width, label, title, metadata):
    fig.text(x, y, label, ha="left", va="center", fontsize=9.4, fontweight="bold", color=INK)
    fig.text(x + 0.030, y, title, ha="left", va="center", fontsize=8.3, color=INK)
    fig.text(
        x + width,
        y,
        metadata,
        ha="right",
        va="center",
        fontsize=6.5,
        color=SECONDARY,
    )
    rule(fig, x, y - 0.021, x + width, y - 0.021)


def panel(fig, image_path: Path, x, y, width, height, time_label, report, report_size=6.7):
    fig.text(
        x,
        y + height + 0.012,
        time_label,
        ha="left",
        va="bottom",
        fontsize=7.2,
        fontweight="bold",
        color=ACCENT,
    )
    radiograph(fig, image_path, x, y, width, height)
    fig.text(
        x,
        y - 0.018,
        report,
        ha="left",
        va="top",
        fontsize=report_size,
        linespacing=1.22,
        color=SECONDARY,
    )


def build(output: Path):
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "pdf.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )
    fig = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT), facecolor="white")

    margin = 0.030
    group_gap = 0.035
    group_width = (1 - 2 * margin - group_gap) / 2
    pair_gap = 0.023
    pair_width = (group_width - pair_gap) / 2

    top_cases = [
        {
            "x": margin,
            "label": "A",
            "title": "Clear interval change",
            "metadata": "20.1 h  ·  horizon 0–24 h  ·  SICU",
            "images": (
                PAIR_SOURCE / "case_a_current_cxr.jpg",
                PAIR_SOURCE / "case_a_observed_followup_cxr.jpg",
            ),
            "times": ("Current  ·  t0", "Observed follow-up  ·  t1"),
            "reports": (
                "Mild cardiomegaly; no edema,\neffusion, pneumonia, or\npneumothorax.",
                "Moderate cardiomegaly with\npulmonary edema; small right\npleural effusion.",
            ),
        },
        {
            "x": margin + group_width + group_gap,
            "label": "B",
            "title": "Stable follow-up",
            "metadata": "24.0 h  ·  horizon 0–24 h  ·  SICU",
            "images": (
                PAIR_SOURCE / "case_b_current_cxr.jpg",
                PAIR_SOURCE / "case_b_observed_followup_cxr.jpg",
            ),
            "times": ("Current  ·  t0", "Observed follow-up  ·  t1"),
            "reports": (
                "Bibasal pleural effusions with\ncompressive atelectasis.",
                "Little interval change; bibasal\neffusions and atelectasis persist.",
            ),
        },
    ]

    header_y = 0.963
    image_y = 0.625
    image_height = 0.275

    for case in top_cases:
        x = case["x"]
        section_header(
            fig,
            x,
            header_y,
            group_width,
            case["label"],
            case["title"],
            case["metadata"],
        )
        right_x = x + pair_width + pair_gap
        panel(
            fig,
            case["images"][0],
            x,
            image_y,
            pair_width,
            image_height,
            case["times"][0],
            case["reports"][0],
        )
        panel(
            fig,
            case["images"][1],
            right_x,
            image_y,
            pair_width,
            image_height,
            case["times"][1],
            case["reports"][1],
        )
        arrow(fig, x + pair_width + 0.004, right_x - 0.004, image_y + image_height / 2)

    rule(fig, 0.500, 0.535, 0.500, 0.982)
    rule(fig, margin, 0.515, 1 - margin, 0.515, width=0.85)

    section_header(
        fig,
        margin,
        0.487,
        1 - 2 * margin,
        "C",
        "Serial follow-up",
        "4 studies  ·  CCU → Vascular → Vascular → Vascular",
    )

    serial_gap = 0.026
    serial_width = (1 - 2 * margin - 3 * serial_gap) / 4
    serial_images = (
        SERIAL_SOURCE / "case_c_t0.jpg",
        SERIAL_SOURCE / "case_c_t1.jpg",
        SERIAL_SOURCE / "case_c_t2.jpg",
        SERIAL_SOURCE / "case_c_t3_observed_followup.jpg",
    )
    serial_times = ("t0", "t1  ·  +9.2 h", "t2  ·  +32.2 h", "t3  ·  +56.3 h")
    serial_reports = (
        "Interstitial edema; small right\neffusion; right pneumothorax\npoorly seen.",
        "Pigtail placed; small apical\npneumothorax persists;\nedema improves.",
        "No pneumothorax after catheter\nrevision; chronic fibrosis\nremains.",
        "No pneumothorax; chronic\nfibrosis; likely small right\neffusion.",
    )
    serial_y = 0.145
    serial_height = 0.285
    serial_x = []
    for index in range(4):
        x = margin + index * (serial_width + serial_gap)
        serial_x.append(x)
        panel(
            fig,
            serial_images[index],
            x,
            serial_y,
            serial_width,
            serial_height,
            serial_times[index],
            serial_reports[index],
            report_size=6.5,
        )

    for index in range(3):
        arrow(
            fig,
            serial_x[index] + serial_width + 0.003,
            serial_x[index + 1] - 0.003,
            serial_y + serial_height / 2,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        format="pdf",
        dpi=300,
        facecolor="white",
        metadata={"Title": "Longitudinal radiograph and report states"},
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Build appendix Figure 1 v3 as a conference-ready PDF.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.output.resolve())


if __name__ == "__main__":
    main()
