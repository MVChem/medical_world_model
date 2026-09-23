"""Real pair schema, lazy source boundary, and task-wide patient holdouts."""
import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from medworld.datasets.protocol import _sha256
from medworld.datasets.temporal import DirectedPairs, TemporalData
from medworld.datasets.temporal_selection import CompactPairs, SELECTION_SCHEMA
from medworld.datasets.unified import UnifiedData


class SelectionFixture:
    def __init__(self, base):
        self.root, self.cxr = base / "selection", base / "cxr"
        self.root.mkdir()
        self.cxr.mkdir()
        self.rows = {split: [] for split in ("train", "validate", "test")}
        for number, (patient, split) in enumerate((("10000001", "train"), ("10000002", "train"),
                                                  ("10000004", "test"))):
            row = {"id": f"medpair:{number:024x}", "patient": patient, "split": split, "hours": 48.5,
                   "full_timeline_adjacent": False,
                   # Deliberately not a usable clinical object: training cannot depend on it.
                   "medication": "not a training input", "evidence_patient": "unavailable"}
            for side, study, time, view in (("source", "50000001", "2180-01-01T00:00:00", "AP"),
                                            ("target", "50000002", "2180-01-03T00:30:00", "PA")):
                dicom = f"image-{patient}-{side}"
                prefix = f"files/p10/p{patient}"
                image = f"{prefix}/s{study}/{dicom}.jpg"
                report = f"{prefix}/s{study}.txt"
                (self.cxr / image).parent.mkdir(parents=True)
                Image.new("L", (8, 12), color=50 if side == "source" else 150).save(self.cxr / image)
                (self.cxr / report).write_text(f"{patient} own {side} report")
                row.update({side: f"cxr:{dicom}", f"{side}_study_id": study, f"{side}_time": time,
                            f"{side}_view": view, f"{side}_image": image, f"{side}_report": report})
            self.rows[split].append(row)
        self.write()

    def write(self):
        outputs = {"evidence.jsonl.gz": {"bytes": 999, "sha256": "missing on purpose"}}
        for split, rows in self.rows.items():
            path = self.root / f"{split}.jsonl.gz"
            # Exercise concatenated gzip members, the actual selection format.
            path.write_bytes(b"".join(gzip.compress((json.dumps(row) + "\n").encode()) for row in rows)
                             or gzip.compress(b""))
            outputs[path.name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        self.manifest = {"schema": SELECTION_SCHEMA, "state": "complete", "dry_run": False,
                         "sources": {"cxr_root": str(self.cxr), "clinical_index": "/does/not/exist",
                                     "clinical_tables": {"hosp.emar": {"path": "/does/not/exist.csv"}}},
                         "outputs": outputs,
                         "cohort": {"splits": {split: {"retained": len(rows)} for split, rows in self.rows.items()},
                                    "retained_pairs": sum(map(len, self.rows.values())),
                                    "retained_patients": len({r["patient"] for rows in self.rows.values() for r in rows})}}
        self.write_manifest()

    def write_manifest(self):
        (self.root / "manifest.json").write_text(json.dumps(self.manifest))


class TemporalSelectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fixture = SelectionFixture(Path(self.tmp.name))

    def test_compact_pairs_need_no_clinical_evidence_or_labels(self):
        with patch("PIL.Image.open", side_effect=AssertionError("Eager image read")):
            data = TemporalData(self.fixture.root)
        self.assertIsInstance(data.pairs["train"], CompactPairs)
        self.assertIsInstance(data.directed("train"), DirectedPairs)
        self.assertEqual(len(data.observations), 6)
        self.assertEqual(set(data.source_hashes), {"manifest.json", "train.jsonl.gz", "validate.jsonl.gz", "test.jsonl.gz"})
        forward, backward = data.directed("train")[:2]
        self.assertEqual(forward["original_id"], self.fixture.rows["train"][0]["id"])
        self.assertEqual((forward["delta_hours"], backward["delta_hours"]), (48.5, -48.5))
        self.assertEqual((forward["source_view"], backward["source_view"]), ("AP", "PA"))
        self.assertEqual(backward["source_time"], forward["target_time"])
        self.assertNotIn("medication", forward)
        self.assertNotIn("evidence_patient", forward)
        batch = data.batch([forward, backward])
        self.assertEqual(set(batch), {"source", "target", "delta_hours"})
        self.assertEqual(batch["source"]["texts"], ["10000001 own source report", "10000001 own target report"])
        self.assertEqual(batch["source"]["images"][0].size, (512, 512))
        self.assertEqual(len(data.directed("validate")), 0)
        self.assertEqual(len(TemporalData(self.fixture.root, bidirectional=False).directed("train")), 2)
        self.assertEqual(data.directed("train")[-1]["direction"], "backward")
        with self.assertRaises(IndexError):
            data.directed("train")[4]

    def test_source_only_never_reads_target_report_or_image(self):
        data = TemporalData(self.fixture.root)
        row = data.directed("train")[0]
        target = data.lookup[row["target"]]
        (self.fixture.cxr / target["report_file"]).unlink()
        (self.fixture.cxr / target["image"]).unlink()
        self.assertEqual(set(data.batch([row], source_only=True)), {"source", "delta_hours"})
        with self.assertRaises(FileNotFoundError):
            data.batch([row])

    def test_pair_fingerprint_and_manifest_state_fail_closed(self):
        path = self.fixture.root / "train.jsonl.gz"
        with path.open("ab") as handle:
            handle.write(gzip.compress(b"\n"))
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            TemporalData(self.fixture.root)
        self.fixture.write()
        for key, value in (("state", "running"), ("dry_run", True)):
            original = self.fixture.manifest[key]
            self.fixture.manifest[key] = value
            self.fixture.write_manifest()
            with self.assertRaisesRegex(ValueError, "complete|dry run"):
                TemporalData(self.fixture.root)
            self.fixture.manifest[key] = original

    def test_ownership_timestamp_and_duplicate_checks(self):
        original = copy.deepcopy(self.fixture.rows)
        cases = [("target_report", "../report.txt"), ("target_image", "/outside/image.jpg"),
                 ("patient", "10000009"), ("split", "test"), ("hours", 0), ("hours", 5),
                 ("hours", float("nan")), ("target_time", "2179-01-01T00:00:00"),
                 ("target_view", "LATERAL"), ("id", "too-long-or-invalid")]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                self.fixture.rows = copy.deepcopy(original)
                self.fixture.rows["train"][0][field] = value
                self.fixture.write()
                with self.assertRaises(ValueError):
                    TemporalData(self.fixture.root)
        self.fixture.rows = copy.deepcopy(original)
        self.fixture.rows["train"].append(copy.deepcopy(original["train"][0]))
        self.fixture.write()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            TemporalData(self.fixture.root)

    def test_cross_split_patient_and_symlink_escape_rejected(self):
        row = dict(self.fixture.rows["train"][0], id="medpair:" + "f" * 24, split="test")
        self.fixture.rows["test"].append(row)
        self.fixture.write()
        with self.assertRaisesRegex(ValueError, "patient split"):
            TemporalData(self.fixture.root)
        self.fixture.rows["test"].pop()
        self.fixture.write()
        path = self.fixture.cxr / self.fixture.rows["train"][0]["source_report"]
        path.unlink()
        outside = self.fixture.root / "outside.txt"
        outside.write_text("wrong report")
        path.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "escapes"):
            TemporalData(self.fixture.root)

    def test_empty_reports_fail_when_used(self):
        data = TemporalData(self.fixture.root)
        row = data.directed("train")[0]
        (self.fixture.cxr / data.lookup[row["source"]]["report_file"]).write_text(" \n")
        with self.assertRaisesRegex(ValueError, "empty report"):
            data.batch([row])

    def test_external_temporal_preserves_vqa_and_classification_holdouts(self):
        from medworld.tests.test_prepared import PreparedDataTests
        prepared = PreparedDataTests()
        prepared.setUp()
        self.addCleanup(prepared.doCleanups)
        patients = {"train": "10000001", "held": "10000002", "test": "10000003"}
        for row in prepared.classification + prepared.segmentation:
            row["subject_id"] = patients.get(row["subject_id"], row["subject_id"])
        extra = dict(prepared.classification[0], id="class-10000004", subject_id="10000004")
        prepared.classification.append(extra)
        path = prepared.root / "vqa/test.json"
        rows = json.loads(path.read_text())
        rows[0]["subject_id"] = "10000002"
        path.write_text(json.dumps(rows))
        prepared.write_manifest()
        data = UnifiedData(dict(prepared.cfg, temporal_data=str(self.fixture.root)))
        self.assertEqual([row["subject_id"] for row in data.rows("classification", "train")], ["10000001"])
        self.assertEqual(len(data.rows("temporal", "train")), 2)
        self.assertEqual(len(data.rows("temporal", "test")), 2)
        self.assertEqual(data.metadata["current_dropped_rows"]["classification/train"], 2)
        self.assertEqual(data.metadata["temporal_excluded_patients"]["train"], 1)
        self.assertFalse(data.metadata["ehr"])
        self.assertTrue(data.metadata["globally_patient_disjoint"])
        self.assertEqual(set(data.training_batch("temporal", 0, 2, 42)), {"source", "target", "delta_hours"})
        self.assertIsInstance(data.temporal.pairs["train"], CompactPairs)


if __name__ == "__main__":
    unittest.main()
