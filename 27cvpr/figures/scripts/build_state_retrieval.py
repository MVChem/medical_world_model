#!/usr/bin/env python3
"""Render a reproducible, CPU-only retrieval pilot from existing visual slots.

Run from any directory with the project Python environment. This script reads
immutable caches, recomputes exact rankings and cohort metrics, and writes only
the generated figure and its audit files. It does not load a model or fit a head.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN = ROOT / "code/medworld_dense_baselines/runs/frozen_slots_20260913"
LABEL_SOURCE = ROOT / "code/medworld_stage1/data/overnight_20260910/observations.jsonl"
FINDINGS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
            "Pleural Effusion", "Pneumothorax"]
FINDING_COLUMNS = [0, 1, 2, 3, 8, 11]
SHORT_NAMES = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
               "Pleural effusion", "Pneumothorax"]
CONDITIONS = {
    "early": {"label": "Early visual depths", "slots": [0, 1], "color": "#327296"},
    "late": {"label": "Late visual depths", "slots": [2, 3], "color": "#947133"},
    "all": {"label": "All four visual depths", "slots": [0, 1, 2, 3], "color": "#49765C"},
}


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def rank_key(identifier, seed):
    return hashlib.sha256(f"{seed}:{identifier}".encode()).hexdigest()


def positive_text(labels):
    names = [name for name, value in zip(SHORT_NAMES, labels) if value == 1]
    return "\n".join(names) if names else "No recorded positive*"


def prepare(args):
    model = args.run / args.model
    contract = read_json(model / "slot_contract.json")
    complete = read_json(model / "slots_complete.json")
    metadata = read_json(model / "slots_model.json")
    data = Path(contract["data_path"])
    assert complete["complete"] and contract["slot_training"] is False
    assert contract["language_model_loaded"] is False
    assert digest(model / "slot_contract.json") == complete["contract_sha256"]
    assert digest(model / "slots_model.json") == complete["model_metadata_sha256"]
    assert digest(model / "hr_slots.npy") == complete["hr_slots_sha256"]
    assert digest(data / "observations.jsonl") == contract["cohort_sha256"]
    assert digest(data / "manifest.json") == contract["manifest_sha256"]
    manifest = read_json(data / "manifest.json")
    assert digest(LABEL_SOURCE) == manifest["sources"][str(LABEL_SOURCE)]
    assert manifest["image_sha256"] == contract["image_sha256"]
    rows = read_rows(data / "observations.jsonl")
    old_rows = read_rows(LABEL_SOURCE)
    ids = np.array([i for i, row in enumerate(rows)
                    if row["split"] == "test" and row["kind"] == "mimic"], dtype=int)
    # Stable ID order also provides a deterministic tie break for cosine ranks.
    ids = np.array(sorted(ids.tolist(), key=lambda i: rows[i]["id"]), dtype=int)
    selected = [rows[i] for i in ids]
    originals = [old_rows[row["old_index"]] for row in selected]
    assert all(row["id"] == old["id"] and row["subject_id"] == old["subject_id"]
               for row, old in zip(selected, originals))
    done = np.load(model / "slots_done.npy")
    assert done[ids, 0].all(), "All test images must have cached HR slots"
    slots = np.load(model / "hr_slots.npy", mmap_mode="r")[ids].astype(np.float32)
    assert slots.shape == (len(ids), 4, 1024) and np.isfinite(slots).all()
    images = np.load(data / "images.npy", mmap_mode="r")
    # Duplicate exclusion is based on exact model-input pixels, beyond patient ID.
    pixel_hashes = [hashlib.sha256(images[i].tobytes()).hexdigest() for i in ids]
    patients = np.array([row["subject_id"] for row in selected])
    pixels = np.array(pixel_hashes)
    allowed = (patients[:, None] != patients[None, :]) & (pixels[:, None] != pixels[None, :])
    assert allowed.sum(1).min() >= 3 and not np.diag(allowed).any()
    labels = np.array([row["labels"] for row in originals], dtype=int)[:, FINDING_COLUMNS]
    positive = labels == 1
    eligible = positive.any(1)
    assert eligible.any()
    # Choice fixed before computing similarities; requires a visible label only.
    query = min(np.flatnonzero(eligible).tolist(),
                key=lambda i: rank_key(selected[i]["id"], args.seed))
    views = np.array([row["view"] for row in originals])
    return dict(contract=contract, complete=complete, metadata=metadata, data=data,
                manifest=manifest, ids=ids, rows=selected, originals=originals,
                slots=slots, images=images, pixel_hashes=pixel_hashes, allowed=allowed,
                labels=labels, positive=positive, eligible=eligible, query=query, views=views)


def retrieval(payload):
    slots = payload["slots"]
    norms = np.linalg.norm(slots, axis=-1, keepdims=True)
    assert (norms > 0).all()
    normalized = slots / norms
    positive = payload["positive"]
    inter = (positive[:, None, :] & positive[None, :, :]).sum(-1)
    union = (positive[:, None, :] | positive[None, :, :]).sum(-1)
    jaccard = np.divide(inter, union, out=np.full(inter.shape, np.nan), where=union > 0)
    view_agree = payload["views"][:, None] == payload["views"][None, :]
    allowed, eligible = payload["allowed"], payload["eligible"]
    # For a uniform random top-3 without replacement, expected mean similarity
    # equals the candidate-bank mean; no Monte Carlo uncertainty is introduced.
    random_per_query = np.nansum(np.where(allowed, jaccard, np.nan), axis=1) / allowed.sum(1)
    random_view_per_query = (view_agree & allowed).sum(1) / allowed.sum(1)
    result = {}
    for key, condition in CONDITIONS.items():
        per_depth = [normalized[:, d] @ normalized[:, d].T for d in condition["slots"]]
        similarity = np.mean(per_depth, axis=0)
        similarity[~allowed] = -np.inf
        ranked = np.argsort(-similarity, axis=1, kind="stable")[:, :3]
        assert allowed[np.arange(len(slots))[:, None], ranked].all()
        qidx = np.arange(len(slots))[:, None]
        overlap = jaccard[qidx, ranked].mean(1)
        view = view_agree[qidx, ranked].mean(1)
        result[key] = dict(indices=ranked, scores=similarity[qidx, ranked],
                           mean_positive_jaccard_at3=float(overlap[eligible].mean()),
                           same_view_at3=float(view.mean()),
                           per_query_jaccard=overlap, per_query_view=view)
    return result, dict(mean_positive_jaccard_at3=float(random_per_query[eligible].mean()),
                        same_view_at3=float(random_view_per_query.mean()))


def plot(payload, result, random, args):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "svg.fonttype": "none", "pdf.fonttype": 42,
                         "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(13.5, 10.8), facecolor="white")
    dark, muted = "#203340", "#576875"
    fig.text(.035, .973, "State-based retrieval: frozen visual-depth pilot",
             color=dark, fontsize=22, weight="bold", va="top")
    fig.text(.035, .934,
             f'{payload["contract"]["model_label"]} vision tower  |  image-only input  |  '
             f'{len(payload["ids"])} test images  |  different-patient retrieval',
             color=muted, fontsize=11.8)
    q = payload["query"]
    q_color = "#263E53"
    fig.text(.042, .883, "Fixed query", fontsize=14, color=q_color, weight="bold")
    fig.text(.301, .883, payload["views"][q], fontsize=11, color=muted, ha="right")
    ax = fig.add_axes([.04, .538, .265, .33125])
    ax.imshow(payload["images"][payload["ids"][q]], cmap="gray", vmin=0, vmax=255)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(q_color); spine.set_linewidth(2)
    fig.text(.044, .517, "Recorded positive findings", fontsize=10.2, color=muted)
    fig.text(.044, .495, positive_text(payload["labels"][q]), fontsize=13.2,
             color=q_color, linespacing=1.5, va="top", weight="bold")
    fig.text(.044, .409, "Full-cohort audit", fontsize=13.2, color=dark, weight="bold")
    fig.text(.044, .385, f'Positive-label Jaccard@3  |  {payload["eligible"].sum()} queries',
             fontsize=9.9, color=muted)
    for line, (key, condition) in enumerate(CONDITIONS.items()):
        y = .357 - .035 * line
        fig.text(.044, y, condition["label"], fontsize=10.1, color=condition["color"])
        fig.text(.292, y, f'{result[key]["mean_positive_jaccard_at3"]:.3f}',
                 fontsize=11, color=condition["color"], ha="right", weight="bold")
    fig.text(.044, .245, "Uniform random bank", fontsize=10.1, color=muted)
    fig.text(.292, .245, f'{random["mean_positive_jaccard_at3"]:.3f}',
             fontsize=11, color=muted, ha="right")
    fig.add_artist(plt.Line2D([.042, .303], [.231, .231], transform=fig.transFigure,
                              color="#CCD5DB", linewidth=.8))
    view_values = " / ".join(f'{100*result[key]["same_view_at3"]:.0f}%' for key in CONDITIONS)
    fig.text(.044, .210, f'Same-view@3, all {len(payload["ids"])} queries',
             fontsize=9.9, color=muted)
    fig.text(.044, .188, f'{view_values}  (early / late / all)', fontsize=10.2, color=dark)
    fig.text(.044, .167, f'Random bank: {100*random["same_view_at3"]:.0f}%', fontsize=10.2, color=muted)
    fig.text(.044, .119, "Scope", fontsize=11, color=dark, weight="bold")
    fig.text(.044, .098, "Four frozen vision summaries (slots 5–8).\n"
             "Fusion slots / full eight-slot state:\nnot available in this cache.",
             fontsize=10.1, color=muted, va="top", linespacing=1.45)
    x_positions = [.350, .565, .780]
    top_positions = [.869, .585, .301]
    for row_index, (key, condition) in enumerate(CONDITIONS.items()):
        top = top_positions[row_index]
        one_based = [payload["metadata"]["block_indices_one_based"][d]
                     for d in condition["slots"]]
        depth_text = ", ".join(map(str, one_based))
        fig.text(.35, top + .014, f'({chr(97+row_index)}) {condition["label"]}',
                 fontsize=13, color=condition["color"], weight="bold")
        fig.text(.975, top + .014, f'Blocks {depth_text}', fontsize=10.2,
                 color=muted, ha="right")
        for rank, x in enumerate(x_positions):
            target = int(result[key]["indices"][q, rank])
            ax = fig.add_axes([x, top - .206, .195, .206])
            ax.imshow(payload["images"][payload["ids"][target]], cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(condition["color"]); spine.set_linewidth(1.2)
            ax.text(.035, .956, f'#{rank+1}', transform=ax.transAxes, ha="left", va="top",
                    color="white", fontsize=10.5, weight="bold",
                    bbox=dict(boxstyle="round,pad=.23", fc=condition["color"], ec="none", alpha=.95))
            ax.text(.97, .955, payload["views"][target], transform=ax.transAxes,
                    ha="right", va="top", color="white", fontsize=9.5,
                    bbox=dict(boxstyle="round,pad=.2", fc="#1F2930", ec="none", alpha=.8))
            score = float(result[key]["scores"][q, rank])
            fig.text(x, top - .222, f'Cosine {score:.4f}', fontsize=8.8, color=muted)
            names = [name for name, value in zip(SHORT_NAMES, payload["labels"][target]) if value == 1]
            # At most three names per line without clipping; missing positives
            # are explicitly described as missing records, not healthy status.
            if len(names) > 2:
                label = ", ".join(names[:2]) + "\n" + ", ".join(names[2:])
            else:
                label = ", ".join(names) if names else "No recorded positive*"
            fig.text(x, top - .238, label, fontsize=8.4, color=dark, va="top", linespacing=1.15)
    fig.text(.350, .017,
             "*Six prespecified findings; missing / uncertain labels are not negatives.  "
             "Rank = mean corresponding-depth cosine.", fontsize=8.7, color=muted)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(args.out / f"state_retrieval.{extension}", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--model", default="qwen08b")
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--out", type=Path, default=ROOT / "27cvpr/figures/generated")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    payload = prepare(args)
    result, random = retrieval(payload)
    plot(payload, result, random, args)
    q = payload["query"]
    per_query = []
    for i, row in enumerate(payload["rows"]):
        record = dict(id=row["id"], subject_id=row["subject_id"],
                      cohort_index=int(payload["ids"][i]), view=payload["views"][i],
                      image_pixel_sha256=payload["pixel_hashes"][i],
                      finding_labels=payload["labels"][i].tolist(),
                      label_metric_eligible=bool(payload["eligible"][i]),
                      candidates=int(payload["allowed"][i].sum()), methods={})
        for key, values in result.items():
            record["methods"][key] = dict(
                positive_jaccard_at3=float(values["per_query_jaccard"][i])
                if payload["eligible"][i] else None,
                same_view_at3=float(values["per_query_view"][i]),
                top3=[dict(id=payload["rows"][int(j)]["id"],
                           subject_id=payload["rows"][int(j)]["subject_id"],
                           score=float(s), view=payload["views"][int(j)],
                           finding_labels=payload["labels"][int(j)].tolist())
                      for j, s in zip(values["indices"][i], values["scores"][i])])
        per_query.append(record)
    provenance = dict(
        status="completed exploratory frozen visual-depth retrieval pilot",
        script=str(Path(__file__).resolve()), script_sha256=digest(__file__),
        command="/home/data2/chk/workspace/2026/.venv/bin/python "
                "27cvpr/figures/scripts/build_state_retrieval.py",
        model=args.model, model_contract=payload["contract"], model_metadata=payload["metadata"],
        slot_cache_sha256=payload["complete"]["hr_slots_sha256"],
        verified_input_hashes=["slot_contract", "slots_model", "hr_slots", "cohort", "manifest", "label_source"],
        image_hash_validation="The source manifest and slot contract agree on full image-cache hash; "
                              "exact prepared-pixel hashes are freshly computed for all test images.",
        label_source=str(LABEL_SOURCE), label_source_sha256=digest(LABEL_SOURCE),
        findings=FINDINGS, finding_source_columns=FINDING_COLUMNS,
        label_encoding={"1": "positive", "0": "negative", "-1": "uncertain", "-2": "missing"},
        protocol=dict(
            image_input="same cached 512x512 HR uint8 radiograph used by native vision slot extractor",
            report_input=False, learned_parameters_in_retrieval=False,
            fusion_slots_present=False, full_eight_slot_state_present=False,
            trained_semantic_slots=False,
            gallery="all MIMIC images in the dense test split; same gallery for every condition",
            exclusions="same subject_id OR identical prepared image pixel SHA256; exact query excluded",
            candidate_patient_uniqueness="query patient excluded; multiple other images of a candidate patient remain eligible",
            ranking="L2-normalize each slot, cosine at corresponding depths, arithmetic mean across selected slots; stable image-ID tie break",
            conditions=CONDITIONS,
            query_selection=f"minimum SHA256('{args.seed}:' + image_id) among test images with >=1 positive of the six findings, before rankings",
            metric="mean of Jaccard(recorded-positive-query-set, recorded-positive-neighbor-set) over the three neighbors, then mean across all eligible queries",
            metric_scope="Queries with at least one recorded positive; blank/uncertain are not interpreted as negatives. Measures recorded positive overlap, not clinical equivalence.",
            random="exact expected mean for three uniformly sampled eligible bank images, without replacement; same exclusions; no fitting",
            confounds="View consistency is reported. Device, exposure, rotation, and anatomy are not controlled; the frozen vision summaries may encode them.",
        ),
        cohort=dict(test_images=len(payload["ids"]), test_patients=len({r["subject_id"] for r in payload["rows"]}),
                    positive_label_queries=int(payload["eligible"].sum()),
                    candidate_counts_min=int(payload["allowed"].sum(1).min()),
                    candidate_counts_max=int(payload["allowed"].sum(1).max()),
                    duplicate_pixel_pairs_across_patients=int(sum(
                        payload["pixel_hashes"][i] == payload["pixel_hashes"][j]
                        and payload["rows"][i]["subject_id"] != payload["rows"][j]["subject_id"]
                        for i in range(len(payload["ids"])) for j in range(i))),
                    view_counts={v: int((payload["views"] == v).sum()) for v in sorted(set(payload["views"]))}),
        metrics={key: {k: v for k, v in values.items() if k in ("mean_positive_jaccard_at3", "same_view_at3")}
                 for key, values in result.items()},
        random_bank_expectation=random, displayed_query=per_query[q],
        per_query_metrics_and_rankings="state_retrieval_rankings.jsonl",
        limitations=["Uses four fixed summaries from the frozen Qwen vision tower, not the learned 4+4 MedWorld state.",
                     "Early/late depth subsets are descriptive probes, not fusion/visual slot-group substitutes.",
                     "Report-derived labels are used only to annotate and audit retrieval, never as state inputs.",
                     "Test selection is a deterministic annotated example, not a best retrieval case; no significance or causal claim."])
    (args.out / "state_retrieval.provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (args.out / "state_retrieval_rankings.jsonl").write_text("\n".join(json.dumps(r) for r in per_query) + "\n")
    print(json.dumps({"outputs": str(args.out), "query_id": payload["rows"][q]["id"],
                      "cohort": provenance["cohort"], "metrics": provenance["metrics"],
                      "random": random}, indent=2))


if __name__ == "__main__":
    main()
