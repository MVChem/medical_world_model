from __future__ import annotations

import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from MIMIC_example.link_mimic_iv_context import build_outputs


def write_gzip_csv(
    path: Path, fieldnames: list[str], rows: list[dict[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class LinkedContextSmokeTest(unittest.TestCase):
    def test_links_transition_to_admission_stay_and_interval_events(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            iv = root / "mimic-iv-3.1"
            hosp, icu = iv / "hosp", iv / "icu"
            subject_id, hadm_id, stay_id = "10000001", "20000001", "30000001"

            write_gzip_csv(
                hosp / "admissions.csv.gz",
                [
                    "subject_id",
                    "hadm_id",
                    "admittime",
                    "dischtime",
                    "edregtime",
                    "edouttime",
                    "admission_type",
                    "admission_location",
                    "discharge_location",
                    "hospital_expire_flag",
                ],
                [
                    {
                        "subject_id": subject_id,
                        "hadm_id": hadm_id,
                        "admittime": "2180-01-01 10:00:00",
                        "dischtime": "2180-01-02 10:00:00",
                        "edregtime": "2180-01-01 08:00:00",
                        "edouttime": "2180-01-01 10:30:00",
                        "admission_type": "EW EMER.",
                        "admission_location": "EMERGENCY ROOM",
                        "discharge_location": "HOME",
                        "hospital_expire_flag": "0",
                    }
                ],
            )
            write_gzip_csv(
                hosp / "transfers.csv.gz",
                [
                    "subject_id",
                    "hadm_id",
                    "transfer_id",
                    "eventtype",
                    "careunit",
                    "intime",
                    "outtime",
                ],
                [
                    {
                        "subject_id": subject_id,
                        "hadm_id": hadm_id,
                        "transfer_id": "1",
                        "eventtype": "admit",
                        "careunit": "Medical Intensive Care Unit (MICU)",
                        "intime": "2180-01-01 08:30:00",
                        "outtime": "2180-01-01 13:00:00",
                    }
                ],
            )
            write_gzip_csv(
                hosp / "diagnoses_icd.csv.gz",
                ["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"],
                [
                    {
                        "subject_id": subject_id,
                        "hadm_id": hadm_id,
                        "seq_num": "1",
                        "icd_code": "J81",
                        "icd_version": "10",
                    }
                ],
            )
            write_gzip_csv(
                hosp / "d_icd_diagnoses.csv.gz",
                ["icd_code", "icd_version", "long_title"],
                [
                    {
                        "icd_code": "J81",
                        "icd_version": "10",
                        "long_title": "Pulmonary edema",
                    }
                ],
            )
            write_gzip_csv(
                hosp / "procedures_icd.csv.gz",
                [
                    "subject_id",
                    "hadm_id",
                    "seq_num",
                    "chartdate",
                    "icd_code",
                    "icd_version",
                ],
                [
                    {
                        "subject_id": subject_id,
                        "hadm_id": hadm_id,
                        "seq_num": "1",
                        "chartdate": "2180-01-01",
                        "icd_code": "5A1D70Z",
                        "icd_version": "10",
                    }
                ],
            )
            write_gzip_csv(
                hosp / "d_icd_procedures.csv.gz",
                ["icd_code", "icd_version", "long_title"],
                [
                    {
                        "icd_code": "5A1D70Z",
                        "icd_version": "10",
                        "long_title": "Performance of urinary filtration",
                    }
                ],
            )
            write_gzip_csv(
                icu / "icustays.csv.gz",
                [
                    "subject_id",
                    "hadm_id",
                    "stay_id",
                    "first_careunit",
                    "last_careunit",
                    "intime",
                    "outtime",
                    "los",
                ],
                [
                    {
                        "subject_id": subject_id,
                        "hadm_id": hadm_id,
                        "stay_id": stay_id,
                        "first_careunit": "MICU",
                        "last_careunit": "MICU",
                        "intime": "2180-01-01 08:30:00",
                        "outtime": "2180-01-01 13:00:00",
                        "los": "0.2",
                    }
                ],
            )
            write_gzip_csv(
                icu / "d_items.csv.gz",
                ["itemid", "label", "category"],
                [
                    {
                        "itemid": "225459",
                        "label": "Chest X-Ray",
                        "category": "5-Imaging",
                    },
                    {
                        "itemid": "225792",
                        "label": "Invasive Ventilation",
                        "category": "2-Ventilation",
                    },
                    {
                        "itemid": "221906",
                        "label": "Norepinephrine",
                        "category": "Medications",
                    },
                ],
            )
            write_gzip_csv(
                icu / "procedureevents.csv.gz",
                [
                    "subject_id",
                    "hadm_id",
                    "stay_id",
                    "starttime",
                    "endtime",
                    "itemid",
                    "value",
                    "valueuom",
                    "location",
                    "statusdescription",
                ],
                [
                    {
                        "subject_id": subject_id,
                        "hadm_id": hadm_id,
                        "stay_id": stay_id,
                        "starttime": "2180-01-01 09:04:00",
                        "endtime": "2180-01-01 09:05:00",
                        "itemid": "225459",
                        "value": "1",
                        "valueuom": "None",
                        "location": "",
                        "statusdescription": "FinishedRunning",
                    },
                    {
                        "subject_id": subject_id,
                        "hadm_id": hadm_id,
                        "stay_id": stay_id,
                        "starttime": "2180-01-01 08:45:00",
                        "endtime": "2180-01-01 12:30:00",
                        "itemid": "225792",
                        "value": "225",
                        "valueuom": "min",
                        "location": "",
                        "statusdescription": "FinishedRunning",
                    },
                ],
            )
            write_gzip_csv(
                icu / "inputevents.csv.gz",
                [
                    "subject_id",
                    "hadm_id",
                    "stay_id",
                    "starttime",
                    "endtime",
                    "itemid",
                    "amount",
                    "amountuom",
                    "rate",
                    "rateuom",
                    "statusdescription",
                ],
                [
                    {
                        "subject_id": subject_id,
                        "hadm_id": hadm_id,
                        "stay_id": stay_id,
                        "starttime": "2180-01-01 10:30:00",
                        "endtime": "2180-01-01 11:00:00",
                        "itemid": "221906",
                        "amount": "3",
                        "amountuom": "mg",
                        "rate": "0.1",
                        "rateuom": "mcg/kg/min",
                        "statusdescription": "FinishedRunning",
                    }
                ],
            )

            packet = {
                "transition_id": "synthetic_transition",
                "source_dataset": "MIMIC-CXR-JPG-2.0.0",
                "patient_id": f"p{subject_id}",
                "current_state": {
                    "study_id": "s50000001",
                    "timestamp": "2180-01-01T09:00:00",
                },
                "interval": {"elapsed_hours": 3.0, "horizon_bin": "0-24h"},
                "future_state": {
                    "study_id": "s50000002",
                    "timestamp": "2180-01-01T12:00:00",
                },
                "curation": {"category": "synthetic"},
            }
            transitions = root / "transitions.jsonl"
            transitions.write_text(json.dumps(packet) + "\n", encoding="utf-8")

            output_dir = root / "linked"
            linked = build_outputs(
                mimic_root=root,
                transitions_path=transitions,
                output_dir=output_dir,
                include_icu_inputs=True,
                render_gallery=True,
            )

            self.assertEqual(linked[0]["linkage_status"], "unique_common_admission")
            self.assertEqual(
                linked[0]["source_databases"],
                {
                    "mimic_cxr_transition": "MIMIC-CXR-JPG-2.0.0",
                    "mimic_iv_linkage_and_context": "MIMIC-IV-3.1",
                },
            )
            context = linked[0]["retrospective_context_for_audit_only"]
            assert isinstance(context, dict)
            self.assertEqual(context["admission"]["hadm_id"], hadm_id)
            self.assertEqual(context["source_location"]["stay"]["stay_id"], stay_id)
            self.assertEqual(
                context["admission_diagnoses"][0]["title"], "Pulmonary edema"
            )
            self.assertEqual(
                context["interval_event_summary"]["procedureevent_counts"],
                {"Chest X-Ray": 1, "Invasive Ventilation": 1},
            )
            self.assertEqual(
                context["interval_event_summary"]["inputevent_counts"],
                {"Norepinephrine": 1},
            )
            self.assertEqual(
                context["timestamp_concordance_checks"]["source_cxr"][
                    "absolute_difference_minutes"
                ],
                4.0,
            )
            self.assertTrue((output_dir / "appendix_cases.csv").is_file())
            self.assertTrue((output_dir / "linked_transitions.jsonl").is_file())
            with (output_dir / "appendix_cases.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                csv_row = next(csv.DictReader(handle))
            self.assertEqual(
                csv_row["cxr_source_database"], "MIMIC-CXR-JPG-2.0.0"
            )
            self.assertEqual(
                csv_row["linkage_context_source_database"], "MIMIC-IV-3.1"
            )
            gallery = (output_dir / "index.html").read_text(encoding="utf-8")
            self.assertIn("DATABASE · MIMIC-CXR-JPG v2.0.0", gallery)
            self.assertIn("DATABASE · MIMIC-IV v3.1", gallery)


if __name__ == "__main__":
    unittest.main()
