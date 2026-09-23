"""Prepare auditable future labels and source-only outcome cohorts for MedWorld.

The original 0923 pair selection is immutable. This writes reference metadata,
human comparison labels, categorical VQA labels, and clinical outcome labels.
It never copies image/report payloads or reads medication tables.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import csv
import zipfile
from bisect import bisect_left

import ijson
import numpy as np

from medworld.datasets.future import FUTURE_TASKS, PROGRESSION_CLASSES, ROW_DTYPE, SCHEMA, SPLITS
from medworld.datasets.protocol import PRIORITY, _sha256, patient_holdouts
from mimic_atlas.patient_index import PatientIndex
from mimic_atlas.build_mimic_transitions import parse_study_datetime
from mimic_atlas.link_mimic_iv_context import admission_start, parse_time

GOLD_RELATIVE = "chest-imagenome-dataset-1.0.0/gold_dataset/gold_object_comparison_with_coordinates.txt"
MORTALITY_SOURCE = "https://physionet.org/content/mimiciv/3.1/"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def source_record(patient, study, dicom, timestamp, split, view):
    base = f"files/p{patient[:2]}/p{patient}"
    return {"id": "cxr:" + dicom, "patient": patient, "study_id": study,
            "timestamp": timestamp.isoformat(), "split": split, "view": view,
            "image": f"{base}/s{study}/{dicom}.jpg", "report_file": f"{base}/s{study}.txt",
            "atlas_url": f"http://127.0.0.1:8767/#subject={patient}"}


def mortality_label(source_time, patient, admissions):
    """Thirty days from acquisition, with date precision and follow-up censoring."""
    horizon = source_time + timedelta(days=30)
    discharges = [parse_time(r.get("dischtime")) for r in admissions]
    discharges = [x for x in discharges if x is not None]
    if not discharges:
        return None, "no_documented_discharge_followup"
    # 365 days is a conservative bound for the documented one-year follow-up.
    censor = max(discharges) + timedelta(days=365)
    if horizon > censor:
        return None, "censored_before_30_days"
    deaths = sorted({parse_time(r.get("deathtime")) for r in admissions if r.get("deathtime")})
    if len(deaths) > 1:
        return None, "conflicting_exact_death_times"
    dod = patient.get("dod", "").strip()
    if deaths:
        death = deaths[0]
        if dod and death.date().isoformat() != dod:
            return None, "death_date_time_disagree"
        if death <= source_time:
            return None, "source_not_before_death"
        return float(death <= horizon), "exact_inpatient_death_time"
    if dod:
        lower = datetime.fromisoformat(dod)
        upper = lower + timedelta(days=1)
        if upper <= source_time:
            return None, "source_after_death_date"
        if lower <= source_time < upper or lower <= horizon < upper:
            return None, "date_only_death_overlaps_boundary"
        return float(source_time < lower and upper <= horizon), "date_only_death_unambiguous"
    return 0., "alive_through_documented_30_day_window"


def has_later_study(source_time, study_id, study_latest):
    """A second projection in the source study is not a future examination."""
    return any(other != study_id and timestamp > source_time
               for other, timestamp in study_latest.items())


def reserve_gold_holdouts(gold, holdouts, seed=42):
    """Reserve about 10% each for validation/test without weakening prior holdouts."""
    support = defaultdict(set)
    for row in gold:
        if holdouts.get(row["patient"], "train") != "human_test":
            support[row["patient"]].add(row["label"])
    target = max(3, round(len(support) * .1))
    assignment = {p: holdouts.get(p, "train") for p in support}
    extras = {}
    def rank(patient, split):
        return hashlib.sha256(f"{seed}:gold:{split}:{patient}".encode()).digest()
    for split in ("test", "validate"):
        selected = {p for p in support if assignment[p] == split}
        while len(selected) < target or set().union(*(support[p] for p in selected)) != {0, 1, 2}:
            candidates = [p for p in support if assignment[p] == "train"]
            if not candidates:
                raise ValueError("Insufficient train patients for disjoint supported gold holdouts")
            covered = set().union(*(support[p] for p in selected))
            chosen = min(candidates, key=lambda p: (-len(support[p] - covered), rank(p, split), p))
            assignment[chosen] = split
            extras[chosen] = split
            selected.add(chosen)
        if len(selected) < 2:
            raise ValueError("Gold holdout requires multiple independent patients")
    return extras, assignment


def gold_comparisons(path, index, cxr_root, audit):
    """Use actual previous/current DICOM references, retaining region semantics."""
    with Path(path).open() as stream:
        raw = list(csv.DictReader(stream, delimiter="\t"))
    needed = defaultdict(set)
    for row in raw:
        needed[row["patient_id"]].update((row["previous_image_id"], row["current_image_id"]))
    observations = {}
    for patient, ids in sorted(needed.items()):
        split_rows = index.read_subject("cxr.mimic-cxr-2.0.0-split", patient)
        split_by_dicom = {r["dicom_id"]: r["split"] for r in split_rows}
        for row in index.read_subject("cxr.mimic-cxr-2.0.0-metadata", patient):
            dicom = row["dicom_id"]
            if dicom not in ids:
                continue
            timestamp = parse_study_datetime(row["StudyDate"], row["StudyTime"])
            if timestamp is None or row["ViewPosition"] not in ("PA", "AP") or dicom not in split_by_dicom:
                audit["gold_excluded_image_metadata_or_view"] += 1
                continue
            obs = source_record(patient, row["study_id"], dicom, timestamp, split_by_dicom[dicom], row["ViewPosition"])
            if not (cxr_root / obs["image"]).is_file() or not (cxr_root / obs["report_file"]).is_file():
                audit["gold_missing_original_asset"] += 1
                continue
            observations[dicom] = obs
    groups = defaultdict(list)
    for row in raw:
        key = (row["patient_id"], row["previous_image_id"], row["current_image_id"], row["bbox"], row["label_name"])
        groups[key].append(row)
    output = []
    classes = {"improved": 0, "no change": 1, "worsened": 2}
    for key, rows in sorted(groups.items()):
        patient, previous, current, region, finding = key
        values = {r["comparison"] for r in rows}
        if len(values) != 1 or next(iter(values)) not in classes:
            audit["gold_mixed_or_conflicting_comparison"] += 1
            continue
        if previous not in observations or current not in observations:
            audit["gold_missing_eligible_exact_endpoint"] += 1
            continue
        source, target = observations[previous], observations[current]
        gap = (datetime.fromisoformat(target["timestamp"]) - datetime.fromisoformat(source["timestamp"])).total_seconds() / 3600
        if gap <= 0 or source["patient"] != patient or target["patient"] != patient:
            audit["gold_invalid_chronology_or_patient"] += 1
            continue
        label = classes[next(iter(values))]
        output.append({"patient": patient, "source": source, "target": target, "hours": gap, "label": label,
            "query": {"question": f"Compared with the source examination, will {finding} in the {region} be improved, stable, or worsened?",
                      "finding": finding, "region": region, "label_source": "Chest ImaGenome human gold comparison",
                      "comparison_relation_ids": sorted({r["relationship_id"] for r in rows}),
                      "comparison_sentences": sorted({r["sentence"] for r in rows}),
                      "answer": PROGRESSION_CLASSES[label]}})
    return output


def silver_comparisons(path, index, holdouts, writer, audit):
    """Train/validate on source-anchored silver relations; gold patients stay test.

    All directions come from existing comparison relations. Equal binary states
    never create a stable label. Test rows are excluded before reading graphs.
    """
    splits = {r["dicom_id"]: r["split"] for r in index.iter_table("cxr.mimic-cxr-2.0.0-split")}
    observations = {}
    for row in index.iter_table("cxr.mimic-cxr-2.0.0-metadata"):
        patient, dicom = row["subject_id"], row["dicom_id"]
        split = holdouts.get(patient, splits.get(dicom))
        if split not in ("train", "validate") or row["ViewPosition"] not in ("PA", "AP"):
            continue
        timestamp = parse_study_datetime(row["StudyDate"], row["StudyTime"])
        if timestamp is not None:
            observations[dicom] = source_record(patient, row["study_id"], dicom, timestamp, split, row["ViewPosition"])
    classes = {"improved": 0, "no change": 1, "worsened": 2}
    with zipfile.ZipFile(path) as archive:
        for number, name in enumerate(archive.namelist()):
            if number % 25000 == 0:
                print("Silver comparison graph entries", number, "/", len(archive.filelist), flush=True)
            if not name.endswith("_SceneGraph.json"):
                continue
            current = Path(name).name.removesuffix("_SceneGraph.json")
            if current not in observations:
                continue
            graph = json.loads(archive.read(name))
            target = observations[current]
            if str(graph["patient_id"]) != target["patient"] or str(graph["study_id"]) != target["study_id"]:
                raise ValueError("Silver graph identity disagrees with original CXR metadata")
            grouped = defaultdict(list)
            for relation in graph.get("relationships", []):
                directions = {n.removeprefix("comparison|yes|") for n in relation.get("relationship_names", [])}
                if len(directions) != 1 or next(iter(directions)) not in classes:
                    audit["silver_mixed_or_unknown_comparison"] += 1
                    continue
                previous = str(relation.get("object_id", "")).split("_", 1)[0]
                actual_current = str(relation.get("subject_id", "")).split("_", 1)[0]
                if actual_current != current or previous not in observations:
                    audit["silver_missing_eligible_exact_endpoint"] += 1
                    continue
                source = observations[previous]
                if source["patient"] != target["patient"] or source["split"] != target["split"]:
                    audit["silver_cross_patient_or_split"] += 1
                    continue
                gap = (datetime.fromisoformat(target["timestamp"]) - datetime.fromisoformat(source["timestamp"])).total_seconds() / 3600
                if gap <= 0:
                    audit["silver_nonpositive_interval"] += 1
                    continue
                label = classes[next(iter(directions))]
                for attribute in relation.get("attributes", []):
                    pieces = attribute.split("|", 2)
                    if len(pieces) != 3 or pieces[0] not in ("anatomicalfinding", "disease", "nlp", "technicalassessment"):
                        continue
                    finding, region = pieces[2], relation["bbox_name"]
                    grouped[(previous, region, finding)].append((label, gap, relation))
            for (previous, region, finding), values in sorted(grouped.items()):
                labels = {value[0] for value in values}
                if len(labels) != 1:
                    audit["silver_conflicting_region_finding_directions"] += 1
                    continue
                label, gap, _ = values[0]
                writer.add("progression", target["split"], observations[previous], target, gap, label,
                    {"question": f"Compared with the source examination, will {finding} in the {region} be improved, stable, or worsened?",
                     "finding": finding, "region": region, "label_source": "Chest ImaGenome source-anchored silver comparison; training/validation only",
                     "comparison_relation_ids": sorted({v[2]["relationship_id"] for v in values}),
                     "source_archive_entry": name,
                     "answer": PROGRESSION_CLASSES[label]})


def vqa_source_candidates(index, holdouts):
    """Index all frontal CXR observations, independently of medication pairs."""
    images, timelines = {}, defaultdict(list)
    for row in index.iter_table("cxr.mimic-cxr-2.0.0-metadata"):
        patient, dicom = row["subject_id"], row["dicom_id"]
        split = holdouts.get(patient)
        if split not in SPLITS or row["ViewPosition"] not in ("PA", "AP"):
            continue
        timestamp = parse_study_datetime(row["StudyDate"], row["StudyTime"])
        if timestamp is None:
            continue
        observation = source_record(patient, row["study_id"], dicom, timestamp, split, row["ViewPosition"])
        images[dicom] = observation
        timelines[patient].append(observation)
    for timeline in timelines.values():
        timeline.sort(key=lambda r: (r["timestamp"], r["view"] == "PA", r["id"]))
    return images, timelines


def closest_prior_vqa_source(target, timeline, cxr_root):
    """Use an earlier distinct study, never another view of the target study."""
    position = bisect_left(timeline, target["timestamp"], key=lambda r: r["timestamp"])
    for source in reversed(timeline[:position]):
        if source["study_id"] == target["study_id"]:
            continue
        if all((cxr_root / source[name]).is_file() for name in ("image", "report_file")):
            return source
    return None


def future_vqa_annotations(vqa_root, index, holdouts, writer, vocabulary, audit):
    """Build a new longitudinal cohort using existing global patient holdouts.

    Official current-image test examples are usually a patient's first image.
    Earlier official train/validation annotations of already held-out patients
    remain legitimate future targets: no task can train on those patients.
    Original annotation partitions are preserved as provenance, not silently
    treated as the new longitudinal split.
    """
    images, timelines = vqa_source_candidates(index, holdouts)
    pairs, source_files, reassignment = {}, {}, Counter()
    for original in ("train", "valid", "test"):
        path = Path(vqa_root) / f"{original}.json"
        source_files[f"vqa_{original}"] = path
        with path.open("rb") as stream:
            for row in ijson.items(stream, "item"):
                patient, study = str(row["subject_id"]), str(row["study_id"])
                split = holdouts.get(patient)
                if len(row["answer"]) != 1:
                    audit["vqa_not_single_answer"] += 1
                    continue
                answer = row["answer"][0].strip().lower()
                if answer not in vocabulary or split not in SPLITS:
                    audit["vqa_not_eligible_target_or_holdout"] += 1
                    continue
                image_id = row["image_id"]
                if image_id not in pairs:
                    target = images.get(image_id)
                    if target is None:
                        pairs[image_id] = None
                    elif target["patient"] != patient or target["study_id"] != study:
                        raise ValueError("Official VQA target identity disagrees with original CXR metadata")
                    elif not all((writer.cxr_root / target[name]).is_file() for name in ("image", "report_file")):
                        pairs[image_id] = None
                    else:
                        source = closest_prior_vqa_source(target, timelines[patient], writer.cxr_root)
                        pairs[image_id] = (source, target) if source is not None else None
                pair = pairs[image_id]
                if pair is None:
                    audit["vqa_no_eligible_prior_or_exact_target"] += 1
                    continue
                source, target = pair
                gap = (datetime.fromisoformat(target["timestamp"]) - datetime.fromisoformat(source["timestamp"])).total_seconds() / 3600
                writer.add("future_vqa", split, source, target, gap, query={
                    "question": row["question"], "answer": answer,
                    "semantic_type": row["semantic_type"], "vqa_id": f"vqa:{original}:{row['idx']}",
                    "vqa_image_id": image_id, "original_annotation_partition": original,
                    "label_source": "Official single-answer exact-target-image CXR VQA",
                    "pair_selection": "Closest earlier eligible frontal image from a distinct study in the full CXR timeline; no medication requirement",
                    "split_policy": "New longitudinal cohort assigned by existing global patient holdout; original current-task rows remain unchanged"})
                reassignment[f"{original}->{split}"] += 1
    return source_files, dict(reassignment)


class Writer:
    def __init__(self, root, cxr_root):
        self.root, self.cxr_root = Path(root), Path(cxr_root)
        self.observations, self.lookup, self.queries = [], {}, []
        self.rows = {task: {split: [] for split in SPLITS} for task in FUTURE_TASKS}

    def observation(self, value, split):
        value = dict(value, split=split)
        identity = value["id"]
        if identity in self.lookup:
            index = self.lookup[identity]
            if self.observations[index] != value:
                raise ValueError(f"Conflicting source observation: {identity}")
            return index
        for field in ("image", "report_file"):
            file = self.cxr_root / value[field]
            if not file.resolve().is_relative_to(self.cxr_root) or not file.is_file():
                raise ValueError("Original future-task asset missing or outside CXR")
        self.lookup[identity] = len(self.observations)
        self.observations.append(value)
        return len(self.observations) - 1

    def add(self, task, split, source, target=None, hours=0., label=0., query=None):
        src = self.observation(source, split)
        tgt = self.observation(target, split) if target else -1
        qid = -1
        if query is not None:
            qid = len(self.queries)
            self.queries.append(query)
        self.rows[task][split].append((src, tgt, qid, hours, label))

    def finish(self):
        for name, rows in (("observations", self.observations), ("queries", self.queries)):
            with (self.root / f"{name}.jsonl").open("w") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        for task in FUTURE_TASKS:
            for split in SPLITS:
                np.save(self.root / f"{task}_{split}.npy", np.asarray(self.rows[task][split], dtype=ROW_DTYPE), allow_pickle=False)


def reuse_progression_labels(root, writer):
    """Reuse only prepared silver labels after a later task's cohort check failed.

    No pixel or feature cache is involved. Exact metadata hashes and the reused
    label provenance are recorded in the completed manifest.
    """
    root = Path(root)
    paths = [root / name for name in ("observations.jsonl", "queries.jsonl", "progression_train.npy", "progression_validate.npy")]
    lineage = {p.name: {"bytes": p.stat().st_size, "sha256": _sha256(p)} for p in paths}
    with paths[0].open() as stream:
        observations = [json.loads(line) for line in stream]
    values = [np.load(root / f"progression_{split}.npy", allow_pickle=False) for split in ("train", "validate")]
    if any(v.dtype != ROW_DTYPE for v in values):
        raise ValueError("Reused progression row dtype mismatch")
    splits = np.concatenate([np.full(len(v), n, dtype=np.uint8) for n, v in enumerate(values)])
    values = np.concatenate(values)
    order = np.argsort(values["query"])
    with paths[1].open() as stream:
        queries = enumerate(stream)
        for count, position in enumerate(order):
            value = values[position]
            while True:
                query_id, line = next(queries)
                if query_id == int(value["query"]):
                    break
                if query_id > int(value["query"]):
                    raise ValueError("Reused progression query ordering mismatch")
            query = json.loads(line)
            if query.get("label_source") != "Chest ImaGenome source-anchored silver comparison; training/validation only":
                raise ValueError("Only previously prepared source-anchored silver labels can be reused")
            split = ("train", "validate")[int(splits[position])]
            source, target = (observations[int(value[name])] for name in ("source", "target"))
            if source["split"] != split or target["split"] != split:
                raise ValueError("Reused progression crosses a patient holdout")
            writer.add("progression", split, source, target, float(value["hours"]), float(value["label"]), query)
            if count % 250000 == 0:
                print("Reused prepared silver labels", count, "/", len(values), flush=True)
    return {"files": lineage, "note": "Reused prepared labels after a future-VQA target-cohort check failed; no images/features cached; original silver exclusion counters were not retained"}


def build(config, output, index_root, *, source_bytes=383, reuse_progression=None):
    from medworld.config import load_config
    from medworld.datasets import UnifiedData
    cfg = load_config(config)
    # Obtain the original reviewed-task/0923 protocol before reserving new patients.
    cfg.update(decoded_image_cache=0, image_workers=2, future_enabled=False)
    root = Path(output).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("Choose an empty prepared future-task output directory")
    root.mkdir(parents=True, exist_ok=True)
    print("Loading original reviewed-task and temporal protocol", flush=True)
    data = UnifiedData(cfg)
    base_fingerprint = data.fingerprint
    holdouts = patient_holdouts(data.current._records, data.temporal.observations)
    index = PatientIndex(index_root)
    cxr_root = Path(index.manifest["cxr_root"])
    gold_path = cxr_root / GOLD_RELATIVE
    audit = Counter()
    gold = gold_comparisons(gold_path, index, cxr_root, audit)
    for row in gold:
        for obs in (row["source"], row["target"]):
            p = obs["patient"]
            if p not in holdouts or PRIORITY[obs["split"]] > PRIORITY[holdouts[p]]:
                holdouts[p] = obs["split"]
    gold_patients = {r["patient"] for r in gold}
    silver_path = cxr_root / "chest-imagenome-dataset-1.0.0/silver_dataset/scene_graph.zip"
    all_gold_test = all(holdouts[p] in ("test", "human_test") for p in gold_patients)
    if all_gold_test:
        extras, gold_assignment = {}, {p: holdouts[p] for p in gold_patients}
        gold_policy = "All284 human gold patients already belong to official VQA test holdouts; keep gold test, use exact-source silver train/validation, never reassign gold into training"
    else:
        extras, gold_assignment = reserve_gold_holdouts(gold, holdouts, cfg["seed"])
        gold_policy = "Seeded patient reservation near80/10/10 with class support; preserve prior test/human_test and drop conflicting current/temporal rows"
        holdouts.update(extras)
    writer = Writer(root, cxr_root)
    for row in gold:
        split = gold_assignment.get(row["patient"])
        if split in SPLITS:
            writer.add("progression", split, row["source"], row["target"], row["hours"], row["label"], row["query"])
    reuse_lineage = None
    if all_gold_test:
        if reuse_progression:
            reuse_lineage = reuse_progression_labels(reuse_progression, writer)
        else:
            silver_comparisons(silver_path, index, holdouts, writer, audit)
    print("Gold labels and new patient holdouts prepared", {s: len(writer.rows['progression'][s]) for s in SPLITS}, flush=True)
    # One label-blind nearest previous selected study per target limits duplicate
    # references without choosing by future findings, question text, or outcomes.
    selected = {}
    for split in SPLITS:
        for pair in data.temporal.pairs[split]:
            patient = str(pair["patient"])
            if holdouts[patient] != split:
                audit["pairs_excluded_new_global_holdout"] += 1
                continue
            target = data.temporal.lookup[pair["target"]]
            key = (patient, target["study_id"])
            rank = (pair["realized_gap_hours"], pair["source"], pair["id"])
            if key not in selected or rank < selected[key][0]:
                selected[key] = (rank, pair, split)

    def temporal_obs(identity):
        row = data.temporal.lookup[identity]
        return source_record(str(row["patient"]), row["study_id"], identity.removeprefix("cxr:"),
                             datetime.fromisoformat(row["timestamp"]), row["split"], row["view"])

    for _, pair, split in selected.values():
        writer.add("future_report", split, temporal_obs(pair["source"]), temporal_obs(pair["target"]),
                   pair["realized_gap_hours"], query={"original_pair_id": pair["id"],
                       "pair_atlas_url": "http://127.0.0.1:8767/#medications&pair=" + pair["id"],
                       "label_source": "Original full future radiology report"})
    vocabulary = json.loads((Path(__file__).resolve().parents[1] / "medworld/datasets/vqa_vocabulary.json").read_text())
    source_files = {"gold_comparisons": gold_path,
                    "study_links": Path(cfg["prepared_data"]) / "study_links.jsonl"}
    if all_gold_test:
        source_files["silver_comparisons"] = silver_path
    vqa_sources, vqa_partitions = future_vqa_annotations(cfg["vqa_data"], index, holdouts, writer, vocabulary, audit)
    source_files.update(vqa_sources)
    print("Paired future report/VQA cohorts prepared", flush=True)
    # Outcomes start from all source studies, including a patient's final exam.
    studies = defaultdict(list)
    with source_files["study_links"].open() as stream:
        for line in stream:
            row = json.loads(line)
            studies[str(row["subject_id"])].append(row)
    outcome_support = {task: {s: Counter() for s in SPLITS} for task in ("mortality_30d", "remaining_los")}
    for number, (patient, timeline) in enumerate(sorted(studies.items())):
        if number % 5000 == 0:
            print("Outcome source patients", number, "/", len(studies), flush=True)
        split = holdouts.get(patient, timeline[0]["effective_holdout"])
        if split not in SPLITS:
            audit["outcomes_excluded_human_holdout"] += len(timeline)
            continue
        patients = index.read_subject("hosp.patients", patient)
        admissions = index.read_subject("hosp.admissions", patient)
        if len(patients) != 1 or not admissions:
            audit["outcomes_missing_patient_or_admission"] += len(timeline)
            continue
        study_latest = {x["study_id"]: datetime.fromisoformat(x["latest_image_timestamp"]) for x in timeline}
        for row in timeline:
            choices = [r for r in row["images"] if r["view"] in ("PA", "AP")]
            if not choices:
                continue
            image = min(choices, key=lambda r: (r["view"] != "PA", r["dicom_id"]))
            source_time = datetime.fromisoformat(image["acquisition_timestamp"])
            matches = []
            for admission in admissions:
                try:
                    start, end = admission_start(admission), parse_time(admission.get("dischtime"))
                except ValueError:
                    continue
                if end is not None and start <= source_time <= end:
                    matches.append(admission)
            if len(matches) != 1:
                audit["outcomes_nonunique_source_admission"] += 1
                continue
            admission = matches[0]
            if image.get("hadm_id") != admission["hadm_id"] or image.get("admission_status") != "unique":
                raise ValueError("Source admission join disagrees with prepared Atlas audit")
            source = source_record(patient, row["study_id"], image["dicom_id"], source_time, split, image["view"])
            if not (cxr_root / source["image"]).is_file() or not (cxr_root / source["report_file"]).is_file():
                audit["outcomes_missing_source_asset"] += 1
                continue
            no_future = not has_later_study(source_time, row["study_id"], study_latest)
            los = (parse_time(admission["dischtime"]) - source_time).total_seconds() / 86400
            common = {"hadm_id": admission["hadm_id"], "has_later_cxr": not no_future,
                      "label_source": "MIMIC-IV 3.1 original patients/admissions"}
            label, reason = mortality_label(source_time, patients[0], admissions)
            audit["mortality:" + reason] += 1
            if label is not None:
                writer.add("mortality_30d", split, source, hours=720., label=label,
                           query={**common, "mortality_label_support": reason})
                outcome_support["mortality_30d"][split]["positive" if label else "negative"] += 1
                outcome_support["mortality_30d"][split]["without_later_cxr"] += int(no_future)
            # Exclude known source-after-death records from LOS as well.
            if reason not in ("source_not_before_death", "source_after_death_date", "death_date_time_disagree", "conflicting_exact_death_times") and los >= 0:
                writer.add("remaining_los", split, source, hours=24., label=los,
                           query={**common, "los_endpoint": "discharge including inpatient death"})
                outcome_support["remaining_los"][split]["rows"] += 1
                outcome_support["remaining_los"][split]["without_later_cxr"] += int(no_future)
    writer.finish()
    counts = {t: {s: len(writer.rows[t][s]) for s in SPLITS} for t in FUTURE_TASKS}
    if any(not n for task in counts.values() for n in task.values()):
        raise ValueError(f"Empty required future-task split: {counts}")
    if any(not outcome_support["mortality_30d"][split][label] for split in SPLITS for label in ("positive", "negative")):
        raise ValueError("Every mortality split requires both positive and negative patients/examples")
    progression_support = {}
    for split in SPLITS:
        values = writer.rows["progression"][split]
        progression_support[split] = {"patients": len({writer.observations[v[0]]["patient"] for v in values}),
            "classes": dict(Counter(PROGRESSION_CLASSES[int(v[4])] for v in values))}
        if set(progression_support[split]["classes"]) != set(PROGRESSION_CLASSES):
            raise ValueError("Progression split lacks a class after eligibility filtering")
    files = {p.name: {"bytes": p.stat().st_size, "sha256": _sha256(p)} for p in sorted(root.iterdir()) if p.is_file()}
    source_files.update(patients=Path(index.table("hosp.patients")["source"]["path"]),
                        admissions=Path(index.table("hosp.admissions")["source"]["path"]))
    manifest = {"schema": SCHEMA, "state": "complete", "created_at": datetime.now(timezone.utc).isoformat(),
        "base_data_fingerprint": base_fingerprint, "base_temporal_manifest_sha256": _sha256(Path(cfg["temporal_data"]) / "manifest.json"),
        "cxr_root": str(cxr_root), "source_report_bytes": source_bytes, "extra_holdouts": extras,
        "counts": counts, "progression_support": progression_support, "outcome_support": outcome_support,
        "vqa_original_partition_to_global_holdout": vqa_partitions,
        "observations": len(writer.observations), "queries": len(writer.queries), "files": files,
        "sources": {k: {"path": str(p.resolve()), "bytes": p.stat().st_size, "sha256": _sha256(p)} for k, p in source_files.items()},
        "implementation_sha256": _sha256(Path(__file__)), "audit": dict(audit), "preparation_reuse": reuse_lineage,
        "rules": {"inputs": "Source image, same UTF-8-bounded original source report, requested horizon, optional task question; no medication/EHR inputs",
            "future_report_pair_selection": "One closest positive incoming 0923 pair per target study; tie-break by source ID/pair ID; no label-dependent selection",
            "future_vqa": "New longitudinal cohort from official exact-target-image annotations across all original partitions, assigned by existing global patient holdout; one normalized answer from fixed official vocabulary; closest earlier eligible frontal source image from a distinct study in the full CXR timeline, independent of medications. Original current-task rows are never relocated",
            "progression": "Chest ImaGenome exact previous/current DICOM, region and finding; human gold evaluation, separately declared source-anchored silver training/validation when gold already held out; reject mixed/conflicting comparisons",
            "gold_holdouts": gold_policy,
            "future_report": "Full original future report file, never the bounded findings/impression export; target only",
            "outcomes": "All eligible source CXR studies uniquely within [min(ED registration, admission), discharge]; no future image or medication selection requirement",
            "mortality_30d": "Death within 30 days of source acquisition; exact inpatient deathtime preferred; ambiguous date-only boundaries excluded; documented last discharge plus conservative 365-day censoring",
            "mortality_documentation": MORTALITY_SOURCE,
            "remaining_los": "(linked discharge - source acquisition)/86400 days; outcome includes discharge by death; fixed input horizon24h, never actual remaining LOS",
            "patient_policy": "Original global holdouts plus declared gold reservations; original task/pair rows are dropped on conflict, never relocated",
            "ehr_inputs": False, "target_images_as_inputs": False}}
    write_json(root / "manifest.json", manifest)
    (root / "README.md").write_text("# MedWorld Table 1 prepared labels\n\n"
        "Source-only future VQA, human progression, full report, 30-day mortality and remaining LOS. "
        "Original images and reports remain referenced in MIMIC-CXR. `manifest.json` records "
        "the exact rules, source hashes, patient reservations and per-task class support.\n\n"
        "Progression split/label provenance is explicit in the manifest and each query. "
        "All human gold patients already in official VQA test remain held out; silver "
        "comparisons may supply training/validation with exact prior-image references. "
        "Outcome cohorts include final source examinations with no later CXR. "
        "No medication or other clinical records are prediction inputs.\n\n"
        "Mortality censoring follows [MIMIC-IV 3.1 documentation](https://physionet.org/content/mimiciv/3.1/).\n")
    print(json.dumps({"output": str(root), "counts": counts, "progression_support": progression_support,
                      "extra_holdout_patients": len(extras), "outcome_support": outcome_support}, indent=2), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--index-root", default="code/mimic_atlas/runs/patient_index_20260918_offsets")
    parser.add_argument("--source-bytes", type=int, default=383)
    parser.add_argument("--reuse-progression", help="Optional prior prepared silver label metadata after another task's cohort check failed")
    args = parser.parse_args()
    if args.source_bytes < 1:
        parser.error("source-bytes must be positive")
    build(args.config, args.output, args.index_root, source_bytes=args.source_bytes, reuse_progression=args.reuse_progression)


if __name__ == "__main__":
    main()
