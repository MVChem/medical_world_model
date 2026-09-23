from datetime import datetime, timedelta
import gzip
import json
from random import Random

import pytest

from data_preprocessing import filter_medicated_pairs as filtering
from data_preprocessing.medication_events import event_overlaps


BASE = datetime(2150, 1, 1)


def test_interval_index_matches_predicate_for_points_and_crossing_infusions():
    rng = Random(73)
    events = []
    for i in range(200):
        lower = rng.randrange(-100, 100)
        upper = lower + rng.choice([0, 0, 1, 20, 50])
        events.append({"subject_id": "123", "start": BASE + timedelta(hours=lower),
                       "end": BASE + timedelta(hours=upper)})
    index = filtering.IntervalIndex(events)
    for lower in range(-90, 90, 3):
        for duration in (1, 5, 24, 200):
            start, end = BASE + timedelta(hours=lower), BASE + timedelta(hours=lower + duration)
            assert index.count(start, end) == sum(event_overlaps(e, "123", start, end) for e in events)
    with pytest.raises(ValueError):
        index.count(BASE, BASE)


def test_histogram_quantiles_interpolate_without_expanding_records():
    result = filtering.summarize_histogram({0: 1, 2: 2, 8: 1})
    assert result["median"] == 2
    assert result["p90"] == pytest.approx(6.2)
    assert result["max"] == 8
    assert result["mean"] == 3


def test_patient_filter_accepts_cross_admission_long_gap_and_different_views(monkeypatch):
    class FakeIndex:
        def read_subject(self, name, subject):
            if name == "hosp.emar":
                return [{"subject_id": subject, "hadm_id": "other-admission", "emar_id": "123-1", "emar_seq": "1",
                         "event_txt": "Administered", "charttime": "2150-01-01 02:00:00", "medication": "Example"}]
            return []
    monkeypatch.setattr(filtering, "_INDEX", FakeIndex())
    monkeypatch.setattr(filtering, "_ITEMS", {})
    observations = [{"patient": "123", "id": f"cxr:{i}", "study_id": str(i),
                     "time": BASE + timedelta(hours=hours), "split": "train", "view": view,
                     "image": f"{i}.jpg", "report": f"{i}.txt", "original_order": i}
                    for i, (hours, view) in enumerate(((0, "AP"), (0.5, "PA"), (24 * 366, "PA")))]
    split, payload, evidence, stats, audit = filtering.process_patient(("123", observations))
    pairs = [json.loads(line) for line in gzip.decompress(payload).splitlines()]
    assert split == "train"
    assert len(pairs) == 2
    assert stats.counts["input_pairs"] == 3
    assert pairs[0]["hours"] > 365 * 24
    assert pairs[0]["source_view"] != pairs[0]["target_view"]
    assert pairs[0]["source_report"] != pairs[0]["target_report"]
    assert json.loads(gzip.decompress(evidence))["records"][0]["hadm_id"] == "other-admission"


def test_continuous_crossing_boundaries_and_missing_dose_distinct():
    events = [{"source_table": "icu.inputevents", "name": "Drug", "event_type": "infusion_segment",
               "start": BASE - timedelta(hours=1), "end": BASE + timedelta(hours=10)},
              {"source_table": "hosp.emar", "name": "drug", "event_type": "administration",
               "start": BASE + timedelta(hours=1), "end": BASE + timedelta(hours=1),
               "dose_details": [{"dose_given": "unknown"}]}]
    result = filtering.EventIndexes(events).query(BASE, BASE + timedelta(hours=2))
    assert result["records"] == 2
    assert result["distinct_recorded_medication_names"] == 1
    assert result["continuous_active_at_source"] == result["continuous_crosses_target"] == 1
    assert result["records_with_positive_numeric_dose"] == 1
    assert result["emar_records_with_detail"] == 1


def test_infusion_starting_at_source_is_active_at_source():
    index = filtering.IntervalIndex([{"start": BASE, "end": BASE + timedelta(hours=3)}])
    assert index.active_at(BASE) == 1
    assert index.crosses(BASE) == 0
    assert index.active_at(BASE + timedelta(hours=3)) == 0
