from __future__ import annotations

import csv
import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

from mimic_atlas.build_mimic_transitions import (
    LABEL_ABSENT,
    LABEL_COLUMNS,
    LABEL_NOT_MENTIONED,
    LABEL_PRESENT,
    Audit,
    BuildConfig,
    Candidate,
    ImageInfo,
    Study,
    build_dataset,
    extract_report_sections,
    find_candidates,
    parse_study_datetime,
    select_candidates,
    state_delta,
    validate_config,
)
from mimic_atlas.forecast_contract import horizon_bin_from_elapsed_hours


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class UnitTests(unittest.TestCase):
    @staticmethod
    def selection_candidate(
        patient_id: str,
        source_id: str,
        target_id: str,
        *,
        flips: int = 0,
        future_positive: int = 0,
    ) -> Candidate:
        timestamp = parse_study_datetime("21800101", "120000")
        assert timestamp is not None
        source_image = ImageInfo(
            dicom_id=f"d{source_id}",
            view="AP",
            timestamp=timestamp,
            image_path=Path(f"/{patient_id}-{source_id}.jpg"),
            relative_path=f"{patient_id}-{source_id}.jpg",
        )
        target_image = ImageInfo(
            dicom_id=f"d{target_id}",
            view="AP",
            timestamp=timestamp + timedelta(hours=48),
            image_path=Path(f"/{patient_id}-{target_id}.jpg"),
            relative_path=f"{patient_id}-{target_id}.jpg",
        )
        labels = tuple(LABEL_ABSENT for _ in LABEL_COLUMNS)
        source = Study(
            subject_id=patient_id,
            study_id=source_id,
            timestamp=timestamp,
            latest_image_timestamp=timestamp,
            report_path=Path(f"/{patient_id}-{source_id}.txt"),
            report_relative_path=f"{patient_id}-{source_id}.txt",
            images_by_view={"AP": source_image},
            split="train",
            labels=labels,
        )
        target = Study(
            subject_id=patient_id,
            study_id=target_id,
            timestamp=timestamp + timedelta(hours=48),
            latest_image_timestamp=timestamp + timedelta(hours=48),
            report_path=Path(f"/{patient_id}-{target_id}.txt"),
            report_relative_path=f"{patient_id}-{target_id}.txt",
            images_by_view={"AP": target_image},
            split="train",
            labels=labels,
        )
        return Candidate(
            source=source,
            target=target,
            source_image=source_image,
            target_image=target_image,
            matched_view="AP",
            source_order=0,
            target_order=1,
            elapsed_hours=48,
            binary_label_flips=flips,
            future_positive_findings=future_positive,
        )

    def test_parse_study_datetime_uses_date_and_fractional_time(self) -> None:
        parsed = parse_study_datetime("21800506", "213014.531")
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(
            parsed.isoformat(timespec="milliseconds"), "2180-05-06T21:30:14.531"
        )
        self.assertIsNone(parse_study_datetime("", "213014"))
        self.assertIsNone(parse_study_datetime("21800506", ""))
        self.assertIsNone(parse_study_datetime("21800506", "not-a-time"))

    def test_extract_report_sections(self) -> None:
        report = "FINAL REPORT\n FINDINGS: Lungs are clear.\n\n IMPRESSION: No acute disease.\n"
        sections = extract_report_sections(report)
        self.assertEqual(sections["findings"], "Lungs are clear.")
        self.assertEqual(sections["impression"], "No acute disease.")
        self.assertIsNone(sections["unsectioned_report"])
        empty = extract_report_sections("FINAL REPORT\nFINDINGS:\nIMPRESSION:\n")
        self.assertTrue(all(value is None for value in empty.values()))

    def test_state_delta_preserves_binary_absent_present_flips(self) -> None:
        current = [LABEL_NOT_MENTIONED] * len(LABEL_COLUMNS)
        future = [LABEL_NOT_MENTIONED] * len(LABEL_COLUMNS)
        atelectasis = LABEL_COLUMNS.index("Atelectasis")
        edema = LABEL_COLUMNS.index("Edema")
        current[atelectasis], future[atelectasis] = LABEL_ABSENT, LABEL_PRESENT
        current[edema], future[edema] = LABEL_PRESENT, LABEL_ABSENT
        delta = state_delta(current, future)
        changes = {
            item["finding"]: item["change"]
            for item in delta["binary_chexpert_label_flips"]
        }
        self.assertEqual(
            changes,
            {"Atelectasis": "changed_to_present", "Edema": "changed_to_absent"},
        )

    def test_coarse_horizon_boundaries(self) -> None:
        cases = {
            24.0: "0-24h",
            24.000001: "24-72h",
            72.0: "24-72h",
            72.000001: "3-7d",
            168.0: "3-7d",
            168.000001: ">7d",
        }
        for elapsed_hours, expected in cases.items():
            with self.subTest(elapsed_hours=elapsed_hours):
                self.assertEqual(
                    horizon_bin_from_elapsed_hours(elapsed_hours), expected
                )

    def test_deterministic_patient_sampling_ignores_future_label_values(self) -> None:
        candidates = [
            self.selection_candidate("10000001", "1", "2", flips=9),
            self.selection_candidate("10000001", "2", "3", flips=0),
            self.selection_candidate("10000002", "4", "5", flips=4),
            self.selection_candidate("10000003", "6", "7", flips=1),
            self.selection_candidate("10000004", "8", "9", flips=7),
        ]
        config = BuildConfig(
            mimic_cxr_root=Path("/unused"),
            output_dir=Path("/unused/out"),
            num_examples=3,
            seed=1729,
        )
        selected = select_candidates(candidates, config, Audit())
        repeated = select_candidates(list(reversed(candidates)), config, Audit())
        self.assertEqual(
            [candidate.source.subject_id for candidate in selected],
            [candidate.source.subject_id for candidate in repeated],
        )
        self.assertEqual(
            [candidate.source.study_id for candidate in selected],
            [candidate.source.study_id for candidate in repeated],
        )

        for index, candidate in enumerate(candidates):
            candidate.binary_label_flips = 100 - index
            candidate.future_positive_findings = index * 10
        relabeled = select_candidates(candidates, config, Audit())
        self.assertEqual(
            [
                (candidate.source.subject_id, candidate.source.study_id)
                for candidate in selected
            ],
            [
                (candidate.source.subject_id, candidate.source.study_id)
                for candidate in relabeled
            ],
        )

        extra_pair = self.selection_candidate("10000001", "10", "11", flips=99)
        with_extra = select_candidates([*candidates, extra_pair], config, Audit())
        self.assertEqual(
            [candidate.source.subject_id for candidate in selected],
            [candidate.source.subject_id for candidate in with_extra],
        )
        self.assertEqual(len({c.source.subject_id for c in selected}), len(selected))

    def test_one_per_patient_rejects_target_conditioned_selection(self) -> None:
        with self.assertRaisesRegex(ValueError, "deterministic_random"):
            validate_config(
                BuildConfig(
                    mimic_cxr_root=Path("/unused"),
                    output_dir=Path("/unused/out"),
                    selection_strategy="change_enriched",
                )
            )
        with self.assertRaisesRegex(ValueError, "min-label-flips 0"):
            validate_config(
                BuildConfig(
                    mimic_cxr_root=Path("/unused"),
                    output_dir=Path("/unused/out"),
                    min_label_flips=1,
                )
            )

    def test_pairing_does_not_skip_an_intervening_incompatible_study(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            labels = tuple(LABEL_ABSENT for _ in LABEL_COLUMNS)

            def make_study(study_id: str, day: int, view: str) -> Study:
                timestamp = parse_study_datetime(f"218001{day:02d}", "120000")
                assert timestamp is not None
                image_path = base / f"{study_id}.jpg"
                report_path = base / f"{study_id}.txt"
                image_path.write_bytes(b"image")
                report_path.write_text("FINDINGS: test", encoding="utf-8")
                image = ImageInfo(
                    dicom_id=f"d{study_id}",
                    view=view,
                    timestamp=timestamp,
                    image_path=image_path,
                    relative_path=image_path.name,
                )
                return Study(
                    subject_id="10000001",
                    study_id=study_id,
                    timestamp=timestamp,
                    latest_image_timestamp=timestamp,
                    report_path=report_path,
                    report_relative_path=report_path.name,
                    images_by_view={view: image},
                    split="train",
                    labels=labels,
                )

            # AP(day 1) -> PA(day 2) -> AP(day 3) has no adjacent AP->AP pair.
            studies = [
                make_study("1", 1, "AP"),
                make_study("2", 2, "PA"),
                make_study("3", 3, "AP"),
            ]
            config = BuildConfig(
                mimic_cxr_root=base,
                output_dir=base / "out",
                num_examples=1,
                view="AP",
                min_label_flips=0,
            )
            self.assertEqual(find_candidates(studies, config, Audit()), [])

            # A random study-ID tie break must not manufacture an order for
            # studies acquired at the same timestamp.
            tied = [
                make_study("4", 4, "AP"),
                make_study("5", 4, "AP"),
                make_study("6", 5, "AP"),
            ]
            self.assertEqual(find_candidates(tied, config, Audit()), [])

            source = make_study("7", 7, "AP")
            target = make_study("8", 8, "AP")
            target.images_by_view["AP"].timestamp = source.timestamp - timedelta(
                hours=1
            )
            self.assertEqual(find_candidates([source, target], config, Audit()), [])


class IntegrationTest(unittest.TestCase):
    def test_builder_orders_by_timestamp_and_keeps_future_text_out_of_inference_inputs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "MIMIC_CXR"
            output = base / "output"
            subject = "10000001"
            studies = [
                (
                    "59000000",
                    "21800102",
                    "120000",
                    "d0",
                    "CURRENT_ONLY normal baseline.",
                ),
                # The numerically smaller study ID is later: IDs must not determine chronology.
                (
                    "51000000",
                    "21800103",
                    "120000",
                    "FUTURE_DICOM_CANARY",
                    "FUTURE_SECRET new opacity.",
                ),
            ]
            metadata_rows = []
            split_rows = []
            label_rows = []
            for index, (study_id, date, time, dicom_id, report_token) in enumerate(
                studies
            ):
                patient_dir = root / "files" / "p10" / f"p{subject}"
                image_dir = patient_dir / f"s{study_id}"
                image_dir.mkdir(parents=True, exist_ok=True)
                (image_dir / f"{dicom_id}.jpg").write_bytes(b"fake-jpeg")
                report = (
                    "FINAL REPORT\n"
                    f" FINDINGS: {report_token}\n\n"
                    f" IMPRESSION: {'No edema.' if index == 0 else 'New edema.'}\n"
                )
                (patient_dir / f"s{study_id}.txt").write_text(report, encoding="utf-8")
                metadata_rows.append(
                    {
                        "dicom_id": dicom_id,
                        "subject_id": subject,
                        "study_id": study_id,
                        "PerformedProcedureStepDescription": "CHEST PORTABLE",
                        "ViewPosition": "AP",
                        "Rows": 1024,
                        "Columns": 1024,
                        "StudyDate": date,
                        "StudyTime": time,
                        "PatientOrientationCodeSequence_CodeMeaning": "Erect",
                    }
                )
                split_rows.append(
                    {
                        "dicom_id": dicom_id,
                        "study_id": study_id,
                        "subject_id": subject,
                        "split": "train",
                    }
                )
                labels = {name: "" for name in LABEL_COLUMNS}
                labels["Edema"] = 0.0 if index == 0 else 1.0
                label_rows.append(
                    {"subject_id": subject, "study_id": study_id, **labels}
                )

            write_csv(
                root / "mimic-cxr-2.0.0-metadata.csv",
                list(metadata_rows[0]),
                metadata_rows,
            )
            write_csv(
                root / "mimic-cxr-2.0.0-split.csv",
                list(split_rows[0]),
                split_rows,
            )
            write_csv(
                root / "mimic-cxr-2.0.0-chexpert.csv",
                ["subject_id", "study_id", *LABEL_COLUMNS],
                label_rows,
            )

            config = BuildConfig(
                mimic_cxr_root=root,
                output_dir=output,
                num_examples=1,
                min_gap_hours=1,
                max_gap_days=5,
                asset_mode="none",
            )
            result = build_dataset(config)
            packet = result.packets[0]
            self.assertEqual(packet["current_state"]["study_id"], "s59000000")
            self.assertEqual(packet["future_state"]["study_id"], "s51000000")
            self.assertIn("CURRENT_ONLY", result.forecast_rows[0]["prompt"])
            self.assertNotIn("FUTURE_SECRET", result.forecast_rows[0]["prompt"])
            self.assertIn(
                "Prediction horizon: 0-24h", result.forecast_rows[0]["prompt"]
            )
            self.assertNotIn("24.0 hours", result.forecast_rows[0]["prompt"])
            self.assertEqual(result.forecast_rows[0]["horizon_bin"], "0-24h")
            self.assertEqual(packet["interval"]["horizon_bin"], "0-24h")

            serialized_input = json.dumps(result.input_rows[0])
            self.assertNotIn("FUTURE_SECRET", serialized_input)
            self.assertNotIn("51000000", serialized_input)
            self.assertNotIn("FUTURE_DICOM_CANARY", serialized_input)
            self.assertNotIn("target_image", result.input_rows[0])
            self.assertNotIn("future_report_for_eval_only", result.input_rows[0])
            self.assertNotIn("elapsed_hours", result.input_rows[0])
            self.assertEqual(result.input_rows[0]["horizon_bin"], "0-24h")
            self.assertIn("FUTURE_SECRET", json.dumps(result.target_rows[0]))
            self.assertIn("FUTURE_SECRET", json.dumps(packet["future_state"]))
            self.assertTrue((output / "index.html").is_file())

            # Transactional reruns replace the entire generated bundle, so a
            # mode change cannot leave stale symlinks or a mismatched summary.
            config.asset_mode = "symlink"
            build_dataset(config)
            assets = list((output / "assets").iterdir())
            self.assertTrue(assets and all(path.is_symlink() for path in assets))
            config.asset_mode = "copy"
            build_dataset(config)
            assets = list((output / "assets").iterdir())
            self.assertTrue(
                assets
                and all(path.is_file() and not path.is_symlink() for path in assets)
            )

            # A no-gallery rerun replaces the full generated bundle and cannot
            # leave stale HTML/assets from an earlier gallery build.
            config.render_gallery = False
            no_gallery = build_dataset(config)
            self.assertNotIn("gallery", no_gallery.output_paths)
            self.assertFalse((output / "index.html").exists())
            self.assertFalse((output / "assets").exists())
            self.assertTrue((output / "forecast_manifest.jsonl").is_file())
            self.assertFalse(no_gallery.summary["gallery_rendered"])

    def test_all_matching_writes_every_pair_without_target_ranked_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "MIMIC_CXR"
            output = base / "all-output"
            study_specs = {
                "10000001": [
                    ("50000001", "21800101", "a1"),
                    ("50000002", "21800102", "a2"),
                    ("50000003", "21800103", "a3"),
                ],
                "10000002": [
                    ("50000004", "21800101", "b1"),
                    ("50000005", "21800102", "b2"),
                ],
            }
            metadata_rows: list[dict[str, object]] = []
            split_rows: list[dict[str, object]] = []
            label_rows: list[dict[str, object]] = []
            for subject, studies in study_specs.items():
                patient_dir = root / "files" / "p10" / f"p{subject}"
                for study_id, date, dicom_id in studies:
                    image_dir = patient_dir / f"s{study_id}"
                    image_dir.mkdir(parents=True, exist_ok=True)
                    (image_dir / f"{dicom_id}.jpg").write_bytes(b"fake-jpeg")
                    (patient_dir / f"s{study_id}.txt").write_text(
                        f"FINAL REPORT\nFINDINGS: current text {study_id}.\n",
                        encoding="utf-8",
                    )
                    metadata_rows.append(
                        {
                            "dicom_id": dicom_id,
                            "subject_id": subject,
                            "study_id": study_id,
                            "PerformedProcedureStepDescription": "CHEST PORTABLE",
                            "ViewPosition": "AP",
                            "Rows": 1024,
                            "Columns": 1024,
                            "StudyDate": date,
                            "StudyTime": "120000",
                            "PatientOrientationCodeSequence_CodeMeaning": "Erect",
                        }
                    )
                    split_rows.append(
                        {
                            "dicom_id": dicom_id,
                            "study_id": study_id,
                            "subject_id": subject,
                            "split": "train",
                        }
                    )
                    labels = {name: 0.0 for name in LABEL_COLUMNS}
                    label_rows.append(
                        {"subject_id": subject, "study_id": study_id, **labels}
                    )

            write_csv(
                root / "mimic-cxr-2.0.0-metadata.csv",
                list(metadata_rows[0]),
                metadata_rows,
            )
            write_csv(
                root / "mimic-cxr-2.0.0-split.csv",
                list(split_rows[0]),
                split_rows,
            )
            label_path = root / "mimic-cxr-2.0.0-chexpert.csv"
            write_csv(
                label_path,
                ["subject_id", "study_id", *LABEL_COLUMNS],
                label_rows,
            )
            config = BuildConfig(
                mimic_cxr_root=root,
                output_dir=output,
                num_examples=None,
                one_per_patient=False,
                min_gap_hours=1,
                max_gap_days=5,
                asset_mode="none",
                render_gallery=False,
                seed=2026,
            )
            first = build_dataset(config)
            first_ids = [row["transition_id"] for row in first.forecast_rows]
            self.assertEqual(len(first_ids), 3)
            self.assertEqual(len(set(first_ids)), 3)
            self.assertEqual(
                sum(row["patient_id"] == "p10000001" for row in first.forecast_rows),
                2,
            )
            self.assertIsNone(first.summary["filters"]["requested_examples"])
            self.assertFalse(first.summary["filters"]["one_per_patient"])
            self.assertFalse(first.summary["selection_policy"]["target_conditioned"])
            for filename in (
                "transitions.jsonl",
                "forecast_manifest.jsonl",
                "forecast_inputs.jsonl",
                "forecast_targets.jsonl",
            ):
                self.assertEqual(
                    len((output / filename).read_text(encoding="utf-8").splitlines()),
                    3,
                )

            # Alter every future-label statistic. Deterministic cohort order
            # and inclusion must remain unchanged.
            for index, row in enumerate(label_rows):
                row["Edema"] = 1.0 if index % 2 else 0.0
                row["Pneumothorax"] = 1.0 if index >= 2 else 0.0
            write_csv(
                label_path,
                ["subject_id", "study_id", *LABEL_COLUMNS],
                label_rows,
            )
            second = build_dataset(config)
            self.assertEqual(
                first_ids,
                [row["transition_id"] for row in second.forecast_rows],
            )


if __name__ == "__main__":
    unittest.main()
