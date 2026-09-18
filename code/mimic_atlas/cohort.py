"""Whole-source patient union and CXR/IV linkage; metadata only, in memory."""

from __future__ import annotations

import csv
import gzip
import logging
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from . import build_mimic_transitions as cxr
from . import link_mimic_iv_context as iv

LOG = logging.getLogger(__name__)


def source_rows(root, relative):
    path = Path(root) / (relative + ".csv.gz")
    if not path.is_file():
        path = path.with_suffix("")
    if not path.is_file():
        raise FileNotFoundError(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def blank_patient(pid):
    return {
        "subject_id": pid,
        "split": None,
        "studies": 0,
        "images": 0,
        "first_time": None,
        "last_time": None,
        "featured": False,
        "has_cxr": False,
        "has_iv": False,
        "admissions": 0,
        "icu_stays": 0,
        "matched_studies": 0,
        "ambiguous_studies": 0,
        "pairs": 0,
        "linked_pairs": 0,
        "gender": None,
        "anchor_age": None,
    }


class CohortIndex:
    def __init__(self):
        self.status = {"state": "loading", "stage": "等待 CXR 目录", "processed": 0}
        self.rows = []
        self.by_id = {}
        self.pairs = []
        self.summary = {}

    def build(self, store):
        """Publish completed data atomically; never read image pixels or reports."""
        try:
            self.status = {
                "state": "loading",
                "stage": "汇总 IV 患者、住院和 ICU",
                "processed": 0,
            }
            records = {
                r["subject_id"]: {
                    **blank_patient(r["subject_id"]),
                    **r,
                    "has_cxr": True,
                }
                for r in store.rows
            }
            warnings = []

            def rows_from(source):
                if getattr(store, "index", None):
                    return store.index.iter_table(source.replace("/", "."))
                return source_rows(store.iv_root, source)

            def patient(pid):
                if pid not in records:
                    records[pid] = blank_patient(pid)
                return records[pid]

            if getattr(store, "index", None):
                for pid in store.index.subjects():
                    patient(pid)
                for pid in store.index.iv_subjects():
                    patient(pid)["has_iv"] = True

            # Patients includes IV-only ED patients without hospital admissions.
            for source in ("hosp/patients", "hosp/admissions", "icu/icustays"):
                try:
                    count = 0
                    for raw in rows_from(source):
                        row = patient(raw["subject_id"])
                        row["has_iv"] = True
                        if source.endswith("patients"):
                            row.update(
                                gender=raw.get("gender"),
                                anchor_age=raw.get("anchor_age"),
                            )
                        elif source.endswith("admissions"):
                            row["admissions"] += 1
                        else:
                            row["icu_stays"] += 1
                        count += 1
                    self.status = {
                        "state": "loading",
                        "stage": f"已汇总 {source}",
                        "processed": count,
                    }
                except FileNotFoundError:
                    warnings.append(f"{source} 缺失；相关覆盖数不完整")
            spans = defaultdict(list)
            try:
                for raw in rows_from("hosp/admissions"):
                    if raw["subject_id"] not in store.by_subject:
                        continue
                    try:
                        start, end = (
                            iv.admission_start(raw),
                            iv.parse_time(raw.get("dischtime")),
                        )
                    except ValueError:
                        continue
                    if end is not None:
                        spans[raw["subject_id"]].append((start, end, raw["hadm_id"]))
            except FileNotFoundError:
                pass

            def matches(pid, t0, t1):
                return [
                    hadm
                    for start, end, hadm in spans[pid]
                    if start <= t0 <= end and start <= t1 <= end
                ]

            config = cxr.BuildConfig(
                store.cxr_root,
                Path("/unused"),
                split="all",
                view="frontal",
                num_examples=None,
                one_per_patient=False,
                render_gallery=False,
                asset_mode="none",
            )
            audit = cxr.Audit()
            pairs, study_counts, labels, views = [], Counter(), Counter(), Counter()
            splits, horizons, flips = {}, Counter(), Counter()
            for n, (pid, studies) in enumerate(store.by_subject.items()):
                if n % 500 == 0:
                    self.status = {
                        "state": "loading",
                        "stage": "全库住院匹配与相邻配对校验",
                        "processed": n,
                        "total": len(store.by_subject),
                    }
                row = records[pid]
                for study in studies:
                    found = matches(pid, study.timestamp, study.timestamp)
                    kind = (
                        "unique"
                        if len(found) == 1
                        else "ambiguous"
                        if found
                        else "unmatched"
                    )
                    study_counts[kind] += 1
                    row["matched_studies"] += kind == "unique"
                    row["ambiguous_studies"] += kind == "ambiguous"
                    views.update(im.view or "UNKNOWN" for im in study.images)
                    if study.labels:
                        labels.update(
                            name
                            for name, val in zip(cxr.LABEL_COLUMNS, study.labels)
                            if val == 1
                        )
                for c in cxr.find_candidates(studies, config, audit):
                    found = matches(
                        pid, c.source_image.timestamp, c.target_image.timestamp
                    )
                    kind = (
                        "unique"
                        if len(found) == 1
                        else "ambiguous"
                        if found
                        else "unmatched"
                    )
                    item = {
                        "subject_id": pid,
                        "transition_id": cxr._transition_id(c),
                        "split": c.source.split,
                        "source_study_id": "s" + c.source.study_id,
                        "target_study_id": "s" + c.target.study_id,
                        "source_time": c.source_image.timestamp.isoformat(),
                        "target_time": c.target_image.timestamp.isoformat(),
                        "view": c.matched_view,
                        "hours": round(c.elapsed_hours, 4),
                        "label_flips": c.binary_label_flips,
                        "linkage": kind,
                        "hadm_id": found[0] if len(found) == 1 else "",
                        "candidate_hadm_ids": found,
                    }
                    pairs.append(item)
                    row["pairs"] += 1
                    row["linked_pairs"] += kind == "unique"
                    stats = splits.setdefault(c.source.split or "unknown", Counter())
                    stats["pairs"] += 1
                    stats["linked_pairs"] += kind == "unique"
                    horizons[
                        "1–24 h"
                        if c.elapsed_hours < 24
                        else "1–3 d"
                        if c.elapsed_hours < 72
                        else "3–7 d"
                        if c.elapsed_hours < 168
                        else "7–30 d"
                        if c.elapsed_hours < 720
                        else "30–365 d"
                    ] += 1
                    flips[
                        "有二元标签变化" if c.binary_label_flips else "无二元标签变化"
                    ] += 1
            rows = sorted(records.values(), key=lambda r: r["subject_id"])
            for row in rows:
                splits.setdefault(row["split"] or "无 CXR split", Counter())[
                    "patients"
                ] += 1
            counts = {
                "patients": len(rows),
                "cxr_patients": sum(r["has_cxr"] for r in rows),
                "iv_patients": sum(r["has_iv"] for r in rows),
                "matched_patients": sum(r["has_cxr"] and r["has_iv"] for r in rows),
                "cxr_only": sum(r["has_cxr"] and not r["has_iv"] for r in rows),
                "iv_only": sum(r["has_iv"] and not r["has_cxr"] for r in rows),
                "admissions": sum(r["admissions"] for r in rows),
                "icu_stays": sum(r["icu_stays"] for r in rows),
                "studies": len(store.studies),
                "images": len(store.images),
                "pairs": len(pairs),
                "paired_patients": sum(r["pairs"] > 0 for r in rows),
                "linked_pairs": sum(r["linked_pairs"] for r in rows),
                "linked_patients": sum(r["linked_pairs"] > 0 for r in rows),
            }
            self.rows, self.by_id, self.pairs = (
                rows,
                records,
                sorted(
                    pairs,
                    key=lambda r: (
                        r["subject_id"],
                        r["source_time"],
                        r["transition_id"],
                    ),
                ),
            )
            interval_bins = [
                {"label": label, "count": 0}
                for label in ("不到 1 天", "1～3 天", "3～7 天", "7～30 天", "30～90 天", "90 天及以上")
            ]
            hours = [pair["hours"] for pair in pairs]
            for value in hours:
                bucket = next((i for i, edge in enumerate((24, 72, 168, 720, 2160)) if value < edge), 5)
                interval_bins[bucket]["count"] += 1
            short_hours = [value for value in hours if value < 24]
            short_bins = [{"label": label, "count": 0} for label in
                          ("1–4 h", "4–8 h", "8–12 h", "12–16 h", "16–20 h", "20–24 h")]
            for value in short_hours:
                bucket = next((i for i, edge in enumerate((4, 8, 12, 16, 20)) if value < edge), 5)
                short_bins[bucket]["count"] += 1
            study_sizes = [len(studies) for studies in store.by_subject.values() if studies]
            study_bins = [{"label": label, "count": 0} for label in
                          ("1 次", "2 次", "3–5 次", "6–10 次", "11–20 次", "21 次及以上")]
            for value in study_sizes:
                bucket = next((i for i, edge in enumerate((1, 2, 5, 10, 20)) if value <= edge), 5)
                study_bins[bucket]["count"] += 1
            self.summary = {
                "counts": counts,
                "splits": splits,
                "study_linkage": study_counts,
                "horizons": horizons,
                "interval_distribution": {
                    "bins": interval_bins,
                    "total": len(hours),
                    "median_days": median(hours) / 24 if hours else None,
                    "within_week_percent": 100 * sum(b["count"] for b in interval_bins[:3]) / len(hours) if hours else None,
                },
                "short_interval_distribution": {
                    "bins": short_bins,
                    "total": len(short_hours),
                    "median_hours": median(short_hours) if short_hours else None,
                },
                "study_count_distribution": {
                    "bins": study_bins,
                    "total": len(study_sizes),
                    "multiple_count": sum(value >= 2 for value in study_sizes),
                    "median_studies": median(study_sizes) if study_sizes else None,
                },
                "label_changes": flips,
                "positive_labels": labels,
                "views": views,
                "audit": audit.as_dict(),
                "warnings": warnings,
                "rules": {
                    "pairing": "严格相邻、同 AP/PA、1 小时至 365 天；校验 split、标签、当前报告及双方图像文件存在",
                    "linkage": "按 subject_id 和实际采集时刻，匹配唯一共同住院（含 ED 起始时间）",
                    "training": "这里是全库可用候选，尚未应用训练任务的抽样、预测契约及实际导出名单；不等于最终训练集。",
                    "files": "配对校验文件存在，未全库解码图像或逐份读取报告。",
                },
            }
            self.status = {
                "state": "ready",
                "stage": "全库匹配完成",
                "processed": len(store.by_subject),
            }
            LOG.info("Cohort ready: %s", counts)
        except Exception as exc:
            LOG.exception("Whole-cohort indexing failed")
            self.status = {"state": "error", "stage": "全库匹配失败", "error": str(exc)}

    def catalog(self):
        return {**self.status, **self.summary}

    def filter_patients(
        self,
        rows,
        query="",
        split="all",
        longitudinal=False,
        coverage="all",
        paired="all",
        sort="subject",
        featured=False,
    ):
        query = query.lower().removeprefix("p")
        result = [
            r
            for r in rows
            if query in r["subject_id"]
            and (split == "all" or (r["split"] or "none") == split)
            and (not longitudinal or r["studies"] > 1)
            and (not featured or r["featured"])
            and (
                coverage == "all"
                or coverage == "cxr"
                and r.get("has_cxr", True)
                or coverage == "matched"
                and r.get("has_cxr")
                and r.get("has_iv")
                or coverage == "cxr_only"
                and r.get("has_cxr")
                and not r.get("has_iv")
                or coverage == "iv_only"
                and r.get("has_iv")
                and not r.get("has_cxr")
            )
            and (
                paired == "all"
                or paired == "pairs"
                and r.get("pairs", 0) > 0
                or paired == "linked"
                and r.get("linked_pairs", 0) > 0
            )
        ]
        if sort != "subject":
            result.sort(key=lambda r: (-r.get(sort, 0), r["subject_id"]))
        return result

    def filter_pairs(self, query="", split="all", linkage="all"):
        return [
            r
            for r in self.pairs
            if query.lower().removeprefix("p") in r["subject_id"]
            and (split == "all" or r["split"] == split)
            and (linkage == "all" or r["linkage"] == linkage)
        ]
