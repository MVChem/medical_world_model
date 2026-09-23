"""Future supervision, observation-time outcomes, and source-only I/O contracts."""
from datetime import datetime, timedelta
import csv
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from PIL import Image
import torch
import numpy as np

from data_preprocessing.build_table12 import Writer, future_vqa_annotations, gold_comparisons, has_later_study, mortality_label, reserve_gold_holdouts, silver_comparisons, source_record
from collections import Counter
from medworld.datasets.future import FutureData, FUTURE_TASKS, SCHEMA, SPLITS, truncate_utf8
from medworld.datasets.protocol import _sha256


class OutcomeLabelsTest(unittest.TestCase):
    def setUp(self):
        self.source = datetime(2180, 1, 1, 12)
        self.admission = {"dischtime": "2180-01-03 12:00:00", "deathtime": ""}

    def test_exact_death_and_censoring(self):
        admission = dict(self.admission, deathtime="2180-01-02 18:00:00")
        self.assertEqual(mortality_label(self.source, {"dod": "2180-01-02"}, [admission])[0], 1.)
        self.assertEqual(mortality_label(self.source, {"dod": ""}, [self.admission])[0], 0.)
        late = self.source + timedelta(days=400)
        self.assertEqual(mortality_label(late, {"dod": ""}, [self.admission])[1], "censored_before_30_days")
        self.assertIsNone(mortality_label(self.source, {"dod": ""}, [dict(admission, deathtime="2180-01-01 11:00:00")])[0])

    def test_death_date_precision_at_both_boundaries(self):
        for date in ("2180-01-01", "2180-01-31"):
            self.assertEqual(mortality_label(self.source, {"dod": date}, [self.admission])[1], "date_only_death_overlaps_boundary")
        self.assertEqual(mortality_label(self.source, {"dod": "2180-01-15"}, [self.admission])[0], 1.)
        self.assertEqual(mortality_label(self.source, {"dod": "2180-02-02"}, [self.admission])[0], 0.)

    def test_future_exam_diagnostic_ignores_later_projection_in_same_study(self):
        latest = {"study1": self.source + timedelta(minutes=2), "older": self.source - timedelta(days=1)}
        self.assertFalse(has_later_study(self.source, "study1", latest))
        latest["study2"] = self.source + timedelta(hours=2)
        self.assertTrue(has_later_study(self.source, "study1", latest))

    def test_reservations_are_disjoint_reproducible_and_preserve_test(self):
        gold = [{"patient": str(10000000 + n), "label": label} for n in range(40) for label in range(3)]
        holds = {str(10000000 + n): "train" for n in range(40)}
        holds["10000000"] = "test"
        holds["10000001"] = "human_test"
        extra, assignment = reserve_gold_holdouts(gold, holds, 42)
        self.assertEqual((extra, assignment), reserve_gold_holdouts(gold, holds, 42))
        self.assertEqual(assignment["10000000"], "test")
        self.assertNotIn("10000000", extra)
        self.assertNotIn("10000001", assignment)
        for split in ("validate", "test"):
            self.assertEqual(sum(value == split for value in assignment.values()), 4)


class FutureFixture:
    def __init__(self, root):
        self.root, self.cxr = root / "prepared", root / "cxr"
        self.root.mkdir(); self.cxr.mkdir()
        writer = Writer(self.root, self.cxr)
        self.holdouts = {}
        self.source_text = "Original source report. " + "患者" * 300
        self.target_text = "Complete future report. " * 150
        for n, split in enumerate(SPLITS):
            patient = str(10000001 + n)
            self.holdouts[patient] = split
            sources = []
            for j in range(2):
                obs = source_record(patient, str(50000001 + j), f"{patient}-image-{j}",
                                    datetime(2180, 1, 1 + j), split, "AP")
                image, report = self.cxr / obs["image"], self.cxr / obs["report_file"]
                image.parent.mkdir(parents=True, exist_ok=True)
                Image.new("L", (8, 10), 50).save(image)
                report.write_text(self.source_text if j == 0 else self.target_text)
                sources.append(obs)
            source, target = sources
            writer.add("future_vqa", split, source, target, 24, query={"question": "Will edema be present?", "answer": "yes", "semantic_type": "verify"})
            writer.add("progression", split, source, target, 24, 2, {"question": "Will left lung edema improve?"})
            writer.add("future_report", split, source, target, 24)
            writer.add("mortality_30d", split, source, hours=720, label=1)
            writer.add("remaining_los", split, source, hours=24, label=2.5)
        writer.finish()
        self.manifest = {"schema": SCHEMA, "state": "complete", "base_data_fingerprint": "original",
            "cxr_root": str(self.cxr), "source_report_bytes": 383, "extra_holdouts": {},
            "counts": {t: {s: 1 for s in SPLITS} for t in FUTURE_TASKS},
            "files": {p.name: {"bytes": p.stat().st_size, "sha256": _sha256(p)} for p in self.root.iterdir()}}
        (self.root / "manifest.json").write_text(json.dumps(self.manifest))
        self.cfg = {"future_data": str(self.root), "future_source_bytes": 383, "image_workers": 1}


class FutureDataTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fixture = FutureFixture(Path(self.tmp.name))
        self.data = FutureData(self.fixture.cfg, "original", self.fixture.holdouts)

    def test_source_only_does_not_open_target_report_or_image(self):
        row = self.data.rows("future_report", "train")[0]
        target = next(o for o in self.data.observations if o["id"] == row["target"])
        (self.fixture.cxr / target["image"]).unlink()
        (self.fixture.cxr / target["report_file"]).unlink()
        batch = self.data.batch("future_report", "train", [0], source_only=True)
        self.assertNotIn("answers", batch)
        self.assertNotIn("labels", batch)
        self.assertEqual(batch["reports"], [truncate_utf8(self.fixture.source_text, 383)])
        self.assertLessEqual(len(batch["reports"][0].encode()), 383)

    def test_full_target_and_task_types(self):
        report = self.data.batch("future_report", "train", [0])
        self.assertEqual(report["answers"], [self.fixture.target_text])
        self.assertGreater(len(report["answers"][0]), 2048)
        vqa = self.data.batch("future_vqa", "train", [0])
        self.assertEqual(vqa["answers"], ["yes"])
        progress = self.data.batch("progression", "train", [0])
        self.assertEqual(progress["labels"].dtype, torch.long)
        self.assertEqual(progress["labels"].tolist(), [2])

    def test_outcomes_require_no_future_and_use_fixed_horizons(self):
        for task, horizon, label in (("mortality_30d", 720, 1), ("remaining_los", 24, 2.5)):
            self.assertNotIn("target", self.data.rows(task, "train")[0])
            batch = self.data.batch(task, "train", [0])
            self.assertEqual(batch["delta_hours"].tolist(), [horizon])
            self.assertEqual(batch["labels"].tolist(), [label])

    def test_protocol_integrity_and_holdouts_are_required(self):
        with self.assertRaisesRegex(ValueError, "different base cohort"):
            FutureData(self.fixture.cfg, "different")
        with self.assertRaisesRegex(ValueError, "holdout"):
            FutureData(self.fixture.cfg, "original", {"10000001": "test"})
        with (self.fixture.root / "queries.jsonl").open("a") as stream:
            stream.write("{}\n")
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            FutureData(self.fixture.cfg, "original")

    def test_declared_pair_horizon_must_match_original_acquisition_times(self):
        path = self.fixture.root / "progression_train.npy"
        values = np.load(path, allow_pickle=False)
        values["hours"] = 1.
        np.save(path, values, allow_pickle=False)
        self.fixture.manifest["files"][path.name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        (self.fixture.root / "manifest.json").write_text(json.dumps(self.fixture.manifest))
        with self.assertRaisesRegex(ValueError, "patient identity or time"):
            FutureData(self.fixture.cfg, "original", self.fixture.holdouts)

    def test_gold_labels_require_explicit_consistent_source_anchored_comparisons(self):
        source, target = self.data.observations[:2]
        previous, current = source["id"].removeprefix("cxr:"), target["id"].removeprefix("cxr:")
        fields = ["patient_id", "previous_image_id", "current_image_id", "bbox", "label_name", "comparison", "relationship_id", "sentence"]
        shared = dict(patient_id=source["patient"], previous_image_id=previous, current_image_id=current,
                      bbox="left lung", label_name="edema", relationship_id="human-gold-1", sentence="Explicit comparison to source.")
        rows = [dict(shared, comparison="no change"),
                dict(shared, label_name="opacity", comparison="improved"),
                dict(shared, label_name="opacity", comparison="worsened"),
                dict(shared, label_name="atelectasis", comparison="no change;;worsened")]
        path = Path(self.tmp.name) / "gold.tsv"
        with path.open("w") as stream:
            writer = csv.DictWriter(stream, fields, delimiter="\t")
            writer.writeheader(); writer.writerows(rows)
        class Index:
            def read_subject(self, table, patient):
                if table.endswith("split"):
                    return [{"dicom_id": i, "split": "train"} for i in (previous, current)]
                return [{"dicom_id": i, "study_id": obs["study_id"], "ViewPosition": "AP", "StudyDate": str(21800101 + n), "StudyTime": "000000"}
                        for n, (i, obs) in enumerate(((previous, source), (current, target))) ]
        audit = Counter()
        result = gold_comparisons(path, Index(), self.fixture.cxr, audit)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["label"], 1)
        self.assertEqual(result[0]["source"]["id"], source["id"])
        self.assertEqual(audit["gold_mixed_or_conflicting_comparison"], 2)

    def test_silver_keeps_exact_prior_reference_and_never_reads_test_graph(self):
        observations = self.data.observations
        class Index:
            def iter_table(self, table):
                if table.endswith("split"):
                    # Global patient holdouts must override this weaker split.
                    return iter({"dicom_id": o["id"].removeprefix("cxr:"), "split": "train"} for o in observations)
                return iter({"dicom_id": o["id"].removeprefix("cxr:"), "subject_id": o["patient"],
                    "study_id": o["study_id"], "ViewPosition": o["view"],
                    "StudyDate": o["timestamp"][:10].replace("-", ""), "StudyTime": "000000"} for o in observations)
        archive_path = Path(self.tmp.name) / "silver.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            for n, split in enumerate(SPLITS):
                source, target = observations[n * 2:n * 2 + 2]
                previous, current = (o["id"].removeprefix("cxr:") for o in (source, target))
                entry = f"scene_graph/{current}_SceneGraph.json"
                if split == "test":
                    archive.writestr(entry, "INVALID: held-out graphs must not be opened")
                    continue
                shared = {"object_id": previous + "_left lung", "subject_id": current + "_left lung",
                          "bbox_name": "left lung", "relationship_id": "exact-prior-relation"}
                relation = lambda finding, names: dict(shared, attributes=["anatomicalfinding|yes|" + finding],
                                                       relationship_names=["comparison|yes|" + name for name in names])
                graph = {"patient_id": target["patient"], "study_id": target["study_id"],
                         "relationships": [relation("edema", ["no change"]),
                            relation("opacity", ["improved"]), relation("opacity", ["worsened"]),
                            relation("atelectasis", ["improved", "worsened"])]}
                archive.writestr(entry, json.dumps(graph))
        writer = Writer(Path(self.tmp.name) / "silver_output", self.fixture.cxr)
        audit = Counter()
        silver_comparisons(archive_path, Index(), self.fixture.holdouts, writer, audit)
        self.assertEqual({s: len(writer.rows["progression"][s]) for s in SPLITS},
                         {"train": 1, "validate": 1, "test": 0})
        self.assertEqual(audit["silver_conflicting_region_finding_directions"], 2)
        self.assertEqual(audit["silver_mixed_or_unknown_comparison"], 2)
        row = writer.rows["progression"]["train"][0]
        self.assertEqual(row[4], 1)
        self.assertEqual(writer.observations[row[0]]["id"], observations[0]["id"])
        self.assertIn("training/validation only", writer.queries[row[2]]["label_source"])

    def test_longitudinal_vqa_uses_global_holdouts_and_preserves_official_provenance(self):
        observations = self.data.observations
        class Index:
            def iter_table(self, table):
                return iter({"dicom_id": o["id"].removeprefix("cxr:"), "subject_id": o["patient"],
                    "study_id": o["study_id"], "ViewPosition": o["view"],
                    "StudyDate": o["timestamp"][:10].replace("-", ""), "StudyTime": "000000"} for o in observations)
        raw = Path(self.tmp.name) / "vqa_raw"
        raw.mkdir()
        annotations = []
        for n, target in enumerate(observations):
            annotations.append({"idx": n, "subject_id": target["patient"], "study_id": target["study_id"],
                "image_id": target["id"].removeprefix("cxr:"), "question": "Will edema be present?",
                "answer": ["yes"], "semantic_type": "verify"})
        # All original annotations were officially train, but these patients
        # already have distinct global holds from other task sources.
        (raw / "train.json").write_text(json.dumps(annotations))
        for partition in ("valid", "test"):
            (raw / f"{partition}.json").write_text("[]")
        writer = Writer(Path(self.tmp.name) / "future_vqa_output", self.fixture.cxr)
        audit = Counter()
        _, partitions = future_vqa_annotations(raw, Index(), self.fixture.holdouts, writer, ["yes"], audit)
        self.assertEqual(partitions, {"train->train": 1, "train->validate": 1, "train->test": 1})
        self.assertEqual(audit["vqa_no_eligible_prior_or_exact_target"], 3)
        for split in SPLITS:
            row = writer.rows["future_vqa"][split][0]
            self.assertEqual(writer.observations[row[0]]["split"], split)
            query = writer.queries[row[2]]
            self.assertEqual(query["original_annotation_partition"], "train")
            self.assertTrue(query["vqa_id"].startswith("vqa:train:"))


if __name__ == "__main__":
    unittest.main()
