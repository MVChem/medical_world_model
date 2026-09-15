#!/usr/bin/env python3
"""Render real frozen-slot spatial outputs and fixed-decoder donor interventions.

No encoder is run, no weights are optimized, and GPU use is disabled. Source
images, masks, slots and final trained decoders all come from the completed
frozen_slots_20260913 experiment. Run with the project Python environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
os.environ["MEDWORLD_PROJECT"] = str(ROOT)
RUN = ROOT / "code/medworld_dense_baselines/runs/frozen_slots_20260913"
sys.path.insert(0, str(RUN / "source"))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from frozen_slots_train import FrozenSlotHead, SlotCorpus, patient_derangement
from common import digest

OUT = ROOT / "27cvpr/figures/generated"
MODEL = "qwen4b"
SEED = 20260913
COLORS = ["#49daf2", "#ffcf58", "#f48cb4"]
PARAMS = {"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9,
          "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
          "figure.facecolor": "white", "savefig.facecolor": "white"}


def rank(row):
    return hashlib.sha256(("candidate-spatial-20260914:" + row["id"]).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def load_head(task, condition):
    mid = "image_only" if condition == "image_only" else MODEL
    path = RUN / mid / f"{task}_{condition}"
    checkpoint = torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == 20
    model = FrozenSlotHead(task).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    contract = read_json(path / "contract.json")
    assert contract == checkpoint["contract"]
    metrics = read_json(path / "metrics.json")
    assert metrics["epochs"] == 20 and metrics["contract_sha256"] == digest(path / "contract.json")
    completion = read_json(RUN / "completion" / f"{mid}_{task}_{condition}.json")
    assert completion["artifact_sha256"] == digest(path / "metrics.json")
    return model, dict(checkpoint=str((path / "checkpoint.pt").relative_to(ROOT)),
                       checkpoint_sha256=digest(path / "checkpoint.pt"),
                       contract=contract, original_reported_metrics=metrics["metrics"])


def measure(task, prediction, target, mask, row):
    if task == "segmentation":
        p = (prediction[:target.shape[0]] > .5) * mask
        t = (target > .5) * mask
        scores = (2 * (p*t).sum((1, 2)) + 1e-8) / (p.sum((1, 2)) + t.sum((1, 2)) + 1e-8)
        return {"dice": float(scores.mean()), "per_organ_dice": scores.tolist()}
    y, x, h, w = row["box"]
    error = prediction[0, y:y+h, x:x+w] - target[0, y:y+h, x:x+w]
    return {"psnr_db": float(-10*np.log10(max(float((error**2).mean()), 1e-12))),
            "mae": float(np.abs(error).mean())}


def prepare():
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    meta = dict(version=1, model="Qwen3.5-4B native vision tower", slots="four frozen visual depth summaries, slots 5-8",
                training="4096 training images, 20 epochs, seed 20260913; final checkpoint",
                world_model_checkpoint=False, jointly_trained_slots=False,
                image_only="same architecture and initial weights, zero slots, separate trained decoder",
                shuffled_trained="same budget and initialization, different-patient conditions during training and evaluation",
                intervention="one normally conditioned decoder per task; its weights and direct image stay fixed; only cached slots are replaced",
                input_protocol="segmentation HR; SR both decoder and slots use the identical cached uint8 LR only; HR is reference only",
                inference="CPU float32; slight differences from original GPU BF16 evaluation are expected",
                selection=dict(spatial="first test and first external-human image in ascending SHA256(candidate-spatial-20260914:<image ID>) order",
                    perturbation_pool="first 16 test images in the same hash order; fixed before inference",
                    perturbation_display="within each task, smallest and largest mean absolute output response to the fixed donor intervention; diagnostic extremes, not reference-error selection",
                    donors="original experiment patient_derangement over the full test split, seed 20260915; never same patient",
                    sr_crop="128x128 HR-coordinate crop, center at 65% valid-ROI width and 58% height; fixed by geometry, never by output"),
                source=str((RUN / "source/frozen_slots_train.py").relative_to(ROOT)),
                source_sha256=digest(RUN / "source/frozen_slots_train.py"),
                slot_contract=read_json(RUN / MODEL / "slot_contract.json"),
                missing="Joint adaptation uses a separate 18708-image/2-epoch protocol and is omitted from this matched comparison.",
                tasks={})
    arrays = {}
    cases = {}
    for task in ("segmentation", "sr"):
        corpus = SlotCorpus(RUN, RUN, MODEL, task, "slots", device="cpu")
        ids = sorted(corpus.pool["test"], key=lambda i: rank(corpus.rows[i]))[:16]
        human = sorted(corpus.pool["human_test"], key=lambda i: rank(corpus.rows[i]))[:1] if task == "segmentation" else []
        donors = patient_derangement(corpus.pool["test"], corpus.rows, SEED + 2)
        if human:
            donors.update(patient_derangement(corpus.pool["human_test"], corpus.rows, SEED+3))
        heads, sources = {}, {}
        for condition in ("image_only", "slots", "shuffled_slots"):
            heads[condition], sources[condition] = load_head(task, condition)
        # Verify matching protocol and initialization instead of assuming it.
        keys = ["train_n", "epochs", "seed", "batch_size", "learning_rate", "cohort_sha256", "split_indices_sha256"]
        assert all(sources[c]["contract"][k] == sources["slots"]["contract"][k] for c in sources for k in keys)
        init_hashes = [read_json(ROOT / v["checkpoint"].replace("checkpoint.pt", "initialization.json"))["state_sha256"] for v in sources.values()]
        assert len(set(init_hashes)) == 1
        cases[task] = {}
        records = []
        for i in ids + human:
            row = corpus.rows[i]
            donor = donors[i]
            assert row["subject_id"] != corpus.rows[donor]["subject_id"]
            image, slots, target, mask = corpus.batch([i])
            donor_slots = torch.from_numpy(np.array(corpus.slots[[donor]], copy=True)).float()
            with torch.inference_mode():
                raw = {"original": heads["slots"](image, slots), "swapped": heads["slots"](image, donor_slots)}
                if i == ids[0] or i in human:
                    raw["image_only"] = heads["image_only"](image, torch.zeros_like(slots))
                    raw["shuffled_trained"] = heads["shuffled_slots"](image, donor_slots)
                if task == "sr":
                    raw["bicubic"] = F.interpolate(image, scale_factor=4, mode="bicubic", align_corners=False)
            target_np = target[0].numpy()
            mask_np = mask[0].numpy()
            predictions = {k: (v.sigmoid() if task == "segmentation" else v.clamp(0, 1))[0].numpy() for k,v in raw.items()}
            assert all(np.isfinite(v).all() for v in predictions.values())
            n_channels = target_np.shape[0]
            difference = np.abs(predictions["original"][:n_channels] - predictions["swapped"][:n_channels]).mean(0)
            response = float((difference * mask_np[0]).sum() / mask_np.sum())
            record = dict(index=i, id=row["id"], subject_id=row["subject_id"], kind=row["kind"], split=row["split"],
                          donor_index=donor, donor_id=corpus.rows[donor]["id"], donor_subject_id=corpus.rows[donor]["subject_id"],
                          mean_absolute_output_change=response, metrics={k: measure(task,v,target_np,mask_np,row) for k,v in predictions.items()})
            case = dict(row=row, record=record, target=target_np, mask=mask_np, predictions=predictions, difference=difference,
                        image=image[0,0].numpy(), donor_image=(corpus.lr_images if task=="sr" else corpus.images)[donor]/255.,
                        hr=corpus.images[i]/255.)
            cases[task][i] = case
            records.append(record)
            for k, v in predictions.items():
                arrays[f"{task}_{i}_{k}"] = v
            arrays[f"{task}_{i}_target"] = target_np
            arrays[f"{task}_{i}_mask"] = mask_np
            if i == ids[0]:
                print(task, "first case", record, flush=True)
        ranked = sorted(ids, key=lambda i:(cases[task][i]["record"]["mean_absolute_output_change"], rank(corpus.rows[i])))
        meta["tasks"][task] = dict(sources=sources, spatial_indices=[ids[0]]+human,
                                  diagnostic_pool_indices=ids, perturbation_indices=[ranked[0],ranked[-1]], cases=records)
        print(task, "intervention range", cases[task][ranked[0]]["record"]["mean_absolute_output_change"], cases[task][ranked[-1]]["record"]["mean_absolute_output_change"], flush=True)
    np.savez_compressed(OUT / "spatial_predictions.npz", **arrays)
    return cases, meta


def base(ax, image):
    ax.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def contours(ax, image, masks, crop=None):
    if crop:
        x,y,w,h = crop
        image = image[y:y+h, x:x+w]
        masks = masks[:,y:y+h,x:x+w]
    base(ax, image)
    for c in range(masks.shape[0]):
        m = masks[c] > .5
        if m.any() and (~m).any():
            ax.contour(m, levels=[.5], colors=[COLORS[c]], linewidths=.8)


def crop_box(case):
    y,x,h,w = case["row"]["box"]
    cx,cy = x+.65*w,y+.58*h
    return int(np.clip(cx-64,0,384)), int(np.clip(cy-64,0,384)),128,128


def crop(image, box):
    x,y,w,h=box
    return image[y:y+h,x:x+w]


def caption(ax, text):
    ax.text(.5,-.055,text,transform=ax.transAxes,ha="center",va="top",fontsize=8.4)


def save(fig, stem):
    for suffix in ("pdf","png","svg"):
        fig.savefig(OUT / f"{stem}.{suffix}", dpi=220, bbox_inches="tight", pad_inches=.08)
    plt.close(fig)


def spatial(cases, meta):
    fig,axes=plt.subplots(4,5,figsize=(10.2,9.8))
    fig.subplots_adjust(left=.09,right=.985,top=.91,bottom=.073,wspace=.055,hspace=.34)
    fig.suptitle("Spatial prediction with frozen visual slots", x=.535,y=.985,fontsize=16,fontweight="bold")
    fig.text(.535,.948,"Qwen3.5-4B visual tower  |  matched 20-epoch decoders  |  4,096 training images",ha="center",fontsize=9.3,color="#444444")
    titles=["Input", "Image-only", "Matched slots", "Shuffled-trained", "Reference"]
    for ax,title in zip(axes[0],titles):ax.set_title(title,pad=9,fontweight="bold")
    seg_ids=meta["tasks"]["segmentation"]["spatial_indices"]
    for r,i in enumerate(seg_ids):
        case=cases["segmentation"][i]; n=case["target"].shape[0]
        base(axes[r,0],case["image"])
        caption(axes[r,0],"MIMIC-CXR" if r==0 else "Montgomery")
        for c,k in enumerate(("image_only","original","shuffled_trained"),1):
            contours(axes[r,c],case["image"],case["predictions"][k][:n]*case["mask"])
            caption(axes[r,c],f'Dice {case["record"]["metrics"][k]["dice"]:.3f}')
        contours(axes[r,4],case["image"],case["target"]*case["mask"])
        caption(axes[r,4],"CXAS pseudo-label" if r==0 else "Human lung masks")
        axes[r,0].text(-.13,.5,"(a) Pseudo reference" if r==0 else "(b) Human reference",transform=axes[r,0].transAxes,rotation=90,ha="center",va="center",fontsize=10,fontweight="bold")
    i=meta["tasks"]["sr"]["spatial_indices"][0];case=cases["sr"][i];box=crop_box(case)
    views=[case["predictions"][k][0] for k in ("bicubic","image_only","original","shuffled_trained")]+[case["target"][0]]
    for c,view in enumerate(views):
        base(axes[2,c],view)
        x,y,w,h=box;axes[2,c].add_patch(Rectangle((x,y),w,h,fill=False,edgecolor="#49daf2",linewidth=1))
        caption(axes[2,c],"Bicubic LR" if c==0 else ("HR, reference only" if c==4 else f'PSNR {case["record"]["metrics"][("image_only","original","shuffled_trained")[c-1]]["psnr_db"]:.2f} dB'))
        base(axes[3,c],crop(view,box))
        if c<4:
            inset=axes[3,c].inset_axes([.63,.02,.35,.35])
            error=np.abs(crop(view-case["target"][0],box))
            inset.imshow(error,cmap="magma",vmin=0,vmax=.10);inset.set_xticks([]);inset.set_yticks([])
            for spine in inset.spines.values():spine.set_color("white");spine.set_linewidth(.7)
    axes[2,0].text(-.13,.5,"(c) 4× super-resolution",transform=axes[2,0].transAxes,rotation=90,ha="center",va="center",fontsize=10,fontweight="bold")
    axes[3,0].text(-.13,.5,"(d) Matched 128 px crop",transform=axes[3,0].transAxes,rotation=90,ha="center",va="center",fontsize=10,fontweight="bold")
    fig.legend([Line2D([0],[0],color=c,lw=1.7) for c in COLORS],["Right lung","Left lung","Heart (pseudo only)"],loc="lower left",bbox_to_anchor=(.085,.012),ncol=3,frameon=False,fontsize=8.5,columnspacing=1)
    fig.text(.65,.031,"Inset: absolute HR error",fontsize=8.5,ha="right")
    cbax=fig.add_axes([.67,.026,.22,.01]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,.1),cmap="magma"),cax=cbax,orientation="horizontal",ticks=[0,.05,.1]);cb.ax.tick_params(labelsize=7,pad=1)
    save(fig,"spatial_outputs")


def perturbation(cases, meta):
    fig,axes=plt.subplots(4,5,figsize=(10.2,10.0))
    fig.subplots_adjust(left=.10,right=.985,top=.902,bottom=.09,wspace=.055,hspace=.38)
    fig.suptitle("Replacing the slot condition across patients", x=.54,y=.985,fontsize=16,fontweight="bold")
    fig.text(.54,.949,"Same decoder + direct image; only four cached visual slots change",ha="center",fontsize=10,color="#333333")
    fig.text(.54,.925,"Smallest / largest response in a fixed 16-image test subset; donor shown as inset",ha="center",fontsize=9,color="#555555")
    titles=["Fixed image / donor", "Original slots", "Donor slots", "Absolute change", "Reference"]
    for ax,title in zip(axes[0],titles):ax.set_title(title,pad=8,fontweight="bold")
    for t,task in enumerate(("segmentation","sr")):
        for j,i in enumerate(meta["tasks"][task]["perturbation_indices"]):
            r=t*2+j;case=cases[task][i]
            label="Smallest response" if j==0 else "Largest response"
            axes[r,0].text(-.13,.5,("Seg. " if task=="segmentation" else "SR ")+label,transform=axes[r,0].transAxes,rotation=90,ha="center",va="center",fontsize=9.8,fontweight="bold")
            base(axes[r,0],case["image"])
            inset=axes[r,0].inset_axes([.60,.02,.38,.38]);base(inset,case["donor_image"])
            for sp in inset.spines.values():sp.set_visible(True);sp.set_color("#49daf2");sp.set_linewidth(1)
            caption(axes[r,0],"Different-patient donor")
            if task=="segmentation":
                for c,k in enumerate(("original","swapped"),1):
                    contours(axes[r,c],case["image"],case["predictions"][k]*case["mask"])
                    caption(axes[r,c],f'Dice {case["record"]["metrics"][k]["dice"]:.3f}')
                contours(axes[r,4],case["image"],case["target"]*case["mask"])
                caption(axes[r,4],"CXAS pseudo-label")
                shown=case["difference"]*case["mask"][0]
                axes[r,3].imshow(shown,cmap="magma",vmin=0,vmax=.25)
            else:
                box=crop_box(case)
                # Input is LR. Rectangle maps the HR crop to its LR coordinates.
                x,y,w,h=box;axes[r,0].add_patch(Rectangle((x/4,y/4),w/4,h/4,fill=False,edgecolor="#ffcf58",linewidth=1))
                for c,k in enumerate(("original","swapped"),1):
                    base(axes[r,c],crop(case["predictions"][k][0],box))
                    caption(axes[r,c],f'PSNR {case["record"]["metrics"][k]["psnr_db"]:.2f} dB')
                base(axes[r,4],crop(case["target"][0],box));caption(axes[r,4],"HR crop, reference only")
                axes[r,3].imshow(crop(case["difference"],box),cmap="magma",vmin=0,vmax=.025)
            axes[r,3].set_xticks([]);axes[r,3].set_yticks([])
            for sp in axes[r,3].spines.values():sp.set_visible(False)
            response=case["record"]["mean_absolute_output_change"]
            caption(axes[r,3],f'Mean |Δ| {response:.5f}')
    fig.text(.103,.05,"Seg.: mean |Δ probability|",fontsize=8.5)
    cax=fig.add_axes([.325,.050,.16,.011]);c=fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,.25),cmap="magma"),cax=cax,orientation="horizontal",ticks=[0,.125,.25]);c.ax.tick_params(labelsize=7,pad=1)
    fig.text(.565,.05,"SR: |Δ intensity|",fontsize=8.5)
    cax=fig.add_axes([.727,.050,.16,.011]);c=fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,.025),cmap="magma"),cax=cax,orientation="horizontal",ticks=[0,.0125,.025]);c.ax.tick_params(labelsize=7,pad=1)
    fig.text(.54,.011,"Mean |Δ| uses the full valid image; SR panels show the same fixed crop. Dark maps preserve weak responses.",ha="center",fontsize=8)
    save(fig,"slot_perturbation")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update(PARAMS)
    cases,meta=prepare()
    spatial(cases,meta);perturbation(cases,meta)
    meta["builder_sha256"]=digest(Path(__file__))
    for stem in ("spatial_outputs","slot_perturbation"):
        (OUT/f"{stem}.json").write_text(json.dumps(meta,indent=2)+"\n")
    print("Wrote spatial_outputs and slot_perturbation PDF, PNG, SVG and JSON",flush=True)


if __name__=="__main__":main()
