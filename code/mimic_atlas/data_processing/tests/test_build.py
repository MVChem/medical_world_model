"""Synthetic end-to-end table joins and safe dataset publication."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from mimic_atlas import build_mimic_transitions as cxr
from mimic_atlas.data_processing.build import build_dataset, parse_args, sha256


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class BuildTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cxr, self.iv = self.root / "cxr", self.root / "iv"
        metadata, splits, labels, admissions = [], [], [], []
        for pid in ("10000001", "10000002"):
            patient = self.cxr / "files/p10" / f"p{pid}"
            for i in range(3):
                sid, dicom = f"{pid}{i}", f"image-{pid}-{i}"
                folder = patient / f"s{sid}"
                folder.mkdir(parents=True)
                Image.new("L", (32, 48), color=70 + i).save(folder / f"{dicom}.jpg")
                (patient / f"s{sid}.txt").write_text("FINDINGS: Clear lungs.\nIMPRESSION: No acute finding.")
                metadata.append(dict(subject_id=pid, study_id=sid, dicom_id=dicom, ViewPosition="AP",
                                     StudyDate=f"2180010{i + 1}", StudyTime="120000", Rows=48, Columns=32))
                splits.append(dict(subject_id=pid, study_id=sid, split="train"))
                labels.append(dict(subject_id=pid, study_id=sid, **{name: 0 for name in cxr.LABEL_COLUMNS}))
            admissions.append(dict(subject_id=pid, hadm_id="a" + pid, admittime="2180-01-01 00:00:00",
                                   dischtime="2180-01-10 00:00:00", edregtime=""))
        write_csv(self.cxr / "mimic-cxr-2.0.0-metadata.csv", list(metadata[0]), metadata)
        write_csv(self.cxr / "mimic-cxr-2.0.0-split.csv", list(splits[0]), splits)
        write_csv(self.cxr / "mimic-cxr-2.0.0-chexpert.csv", list(labels[0]), labels)
        write_csv(self.iv / "hosp/admissions.csv", list(admissions[0]), admissions)
        write_csv(self.iv / "hosp/patients.csv", ["subject_id"], [{"subject_id": pid} for pid in ("10000001", "10000002")])
        write_csv(self.iv / "icu/icustays.csv", ["subject_id", "hadm_id", "stay_id", "intime", "outtime"], [])
        self.args = parse_args(["--cxr-root", str(self.cxr), "--iv-root", str(self.iv),
                               "--output-dir", str(self.root / "out")])

    @staticmethod
    def supervision(output, *_):
        (output / "segmentation.jsonl").write_text("")
        return {"summary": {"counts": {"segmentation": {"train": 0}, "vqa": {"test": 1}}},
                "holdouts": {"10000002": "test"}}

    def test_full_build_excludes_holdout_and_survives_rename(self):
        with patch("mimic_atlas.data_processing.build.export_reviewed_supervision", self.supervision):
            manifest = build_dataset(self.args)
        root = self.args.output_dir
        self.assertEqual(manifest["counts"]["classification"]["train"], 3)
        self.assertEqual(manifest["counts"]["temporal"]["train"], 3)
        self.assertEqual(manifest["pair_kinds"], {"adjacent": 2, "nonadjacent": 1})
        self.assertEqual(manifest["counts"]["segmentation"], {"train": 0})
        self.assertTrue((root / "images").is_symlink())
        observations = [json.loads(line) for line in (root / "temporal/observations.jsonl").read_text().splitlines()]
        self.assertEqual(len(observations), 3)
        self.assertTrue(all((root / "temporal" / row["image"]).is_file() for row in observations))
        self.assertTrue(all(row["patient"] == "10000001" and len(row["labels"]) == 13 for row in observations))
        for name, digest in manifest["file_sha256"].items():
            self.assertEqual(sha256(root / name), digest)
        with self.assertRaises(FileExistsError):
            build_dataset(self.args)

    def test_failed_build_cleans_staging_and_never_publishes(self):
        with patch("mimic_atlas.data_processing.build.export_reviewed_supervision", side_effect=ValueError("broken source")):
            with self.assertRaisesRegex(ValueError, "broken source"):
                build_dataset(self.args)
        self.assertFalse(self.args.output_dir.exists())
        self.assertEqual(list(self.root.glob(".out-*")), [])

    def test_same_seed_produces_identical_data_rows(self):
        with patch("mimic_atlas.data_processing.build.export_reviewed_supervision", self.supervision):
            first = build_dataset(self.args)
            self.args.output_dir = self.root / "second"
            second = build_dataset(self.args)
        self.assertEqual(first["file_sha256"], second["file_sha256"])

    def test_replace_through_symlink_keeps_alias_and_archives_prior_bundle(self):
        central = self.root / "central"
        central.mkdir()
        (central / ".medworld-prepared").write_text("medworld-prepared-v1\n")
        (central / "prior.txt").write_text("prior bundle")
        self.args.output_dir.symlink_to(central, target_is_directory=True)
        self.args.replace = True
        with patch("mimic_atlas.data_processing.build.export_reviewed_supervision", self.supervision):
            build_dataset(self.args)
        self.assertTrue(self.args.output_dir.is_symlink())
        self.assertTrue((self.args.output_dir / "manifest.json").is_file())
        backups = list(self.root.glob("central_previous_*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual((backups[0] / "prior.txt").read_text(), "prior bundle")

    def test_failed_publish_restores_previous_bundle(self):
        self.args.output_dir.mkdir()
        (self.args.output_dir / ".medworld-prepared").write_text("medworld-prepared-v1\n")
        (self.args.output_dir / "prior.txt").write_text("prior bundle")
        self.args.replace = True
        rename = Path.rename

        def failing_rename(path, target):
            if path.name.startswith(".out-"):
                raise OSError("publication failed")
            return rename(path, target)

        with patch("mimic_atlas.data_processing.build.export_reviewed_supervision", self.supervision), \
                patch.object(Path, "rename", failing_rename):
            with self.assertRaisesRegex(OSError, "publication failed"):
                build_dataset(self.args)
        self.assertEqual((self.args.output_dir / "prior.txt").read_text(), "prior bundle")
        self.assertEqual(list(self.root.glob("out_previous_*")), [])
        self.assertEqual(list(self.root.glob(".out-*")), [])


if __name__ == "__main__":
    unittest.main()
