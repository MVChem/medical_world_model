from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from mimic_atlas.build_mimic_transitions import ImageInfo, LABEL_COLUMNS, Study
from mimic_atlas.data_processing.pairing import select_patient_pairs


BASE = datetime(2100, 1, 1)


def study(number, hours, *, view="AP", split="train", subject="10000001"):
    timestamp = BASE + timedelta(hours=hours)
    image = ImageInfo(str(number), view, timestamp, Path("/unused.jpg"), "unused.jpg")
    return Study(
        subject, str(number), timestamp, timestamp, Path("/unused.txt"), "unused.txt",
        images_by_view={view: image}, split=split,
        labels=(0,) * len(LABEL_COLUMNS), images=[image],
    )


def admission(start=-1, end=1000, *, hadm="1", subject="10000001"):
    return {
        "subject_id": subject, "hadm_id": hadm,
        "admittime": (BASE + timedelta(hours=start)).isoformat(),
        "dischtime": (BASE + timedelta(hours=end)).isoformat(),
    }


def identities(rows):
    return [(row["source_study"].study_id, row["target_study"].study_id) for row in rows]


def test_strict_adjacency_retains_ineligible_intermediate_study():
    timeline = [study(1, 0), study(2, 24, view="LATERAL"), study(3, 48)]
    rows, _ = select_patient_pairs(timeline, [admission()], mode="adjacent")
    assert rows == []
    rows, _ = select_patient_pairs(timeline, [admission()], mode="adjacent_random")
    assert identities(rows) == [("1", "3")]
    assert rows[0]["kind"] == "nonadjacent"
    assert (rows[0]["source_order"], rows[0]["target_order"]) == (0, 2)
    rows, _ = select_patient_pairs(
        [study(1, 0), study(2, 24), study(3, 48)], [admission()],
        mode="adjacent", eligible_study_ids={"1", "3"},
    )
    assert rows == []


def test_random_is_bounded_reproducible_and_label_value_independent():
    timeline = [study(n, n * 24) for n in range(12)]
    kwargs = dict(mode="adjacent_random", seed=731, random_pairs_per_patient=4)
    rows, audit = select_patient_pairs(timeline, [admission()], **kwargs)
    assert audit["selected_adjacent_pairs"] == 11
    assert audit["selected_nonadjacent_pairs"] == 4
    reverse, _ = select_patient_pairs(reversed(timeline), [admission()], **kwargs)
    assert identities(rows) == identities(reverse)
    for item in timeline:
        item.labels = (1,) * len(LABEL_COLUMNS)
    changed, _ = select_patient_pairs(timeline, [admission()], **kwargs)
    assert identities(rows) == identities(changed)
    another_seed, _ = select_patient_pairs(timeline, [admission()], **(kwargs | {"seed": 732}))
    assert identities(rows) != identities(another_seed)
    assert len(set(identities(rows))) == len(rows)


def test_all_and_random_modes_have_explicit_nonadjacent_scope():
    timeline = [study(n, n * 24) for n in range(5)]
    rows, audit = select_patient_pairs(timeline, [admission()], mode="all")
    assert len(rows) == 10
    assert audit["selected_adjacent_pairs"] == 4
    rows, _ = select_patient_pairs(timeline, [admission()], mode="random", random_pairs_per_patient=50)
    assert len(rows) == 6
    assert all(row["kind"] == "nonadjacent" for row in rows)
    rows, _ = select_patient_pairs(timeline, [admission()], random_pairs_per_patient=0)
    assert len(rows) == 4


def test_selected_image_timestamps_and_ed_start_determine_match_and_gap():
    source, target = study(1, 0), study(2, 48)
    source.images_by_view["AP"].timestamp = BASE + timedelta(hours=2)
    source.latest_image_timestamp = BASE + timedelta(hours=2)
    hospital = admission(3, 50)
    hospital["edregtime"] = (BASE + timedelta(hours=1)).isoformat()
    rows, _ = select_patient_pairs([source, target], [hospital])
    assert len(rows) == 1
    assert rows[0]["realized_gap_hours"] == 46
    assert rows[0]["hadm_id"] == "1"


def test_each_endpoint_must_match_unambiguously_even_with_unique_common_admission():
    rows, audit = select_patient_pairs(
        [study(1, 0), study(2, 48)], [admission(), admission(-1, 12, hadm="2")]
    )
    assert rows == []
    assert audit["pairs_rejected_ambiguous_admission"] == 1


def test_exact_subject_and_admission_membership():
    timeline = [study(1, 0), study(2, 48)]
    assert select_patient_pairs(timeline, [admission(subject="10000002")])[0] == []
    separate = [admission(-1, 12), admission(24, 72, hadm="2")]
    assert select_patient_pairs(timeline, separate)[0] == []
    rows, _ = select_patient_pairs(timeline, separate, linkage="patient")
    assert len(rows) == 1
    assert rows[0]["hadm_id"] is None
    with pytest.raises(ValueError, match="one patient"):
        select_patient_pairs([study(1, 0), study(2, 48, subject="10000002")], [])


def test_gap_bounds_are_inclusive_and_patient_splits_cannot_mix():
    assert len(select_patient_pairs([study(1, 0), study(2, 1)], [admission()])[0]) == 1
    assert len(select_patient_pairs([study(1, 0), study(2, 24)], [admission()], max_gap_days=1)[0]) == 1
    assert select_patient_pairs([study(1, 0), study(2, 24.01)], [admission()], max_gap_days=1)[0] == []
    with pytest.raises(ValueError, match="split violation"):
        select_patient_pairs([study(1, 0), study(2, 24, split="test")], [admission()])
    assert select_patient_pairs([study(1, 0), study(2, 24, split=None)], [admission()])[0] == []


def test_tied_and_overlapping_acquisition_windows_are_rejected():
    timeline = [study(1, 0), study(2, 0), study(3, 24)]
    assert select_patient_pairs(timeline, [admission()], mode="all")[0] == []
    source, target = study(1, 0), study(2, 24)
    source.latest_image_timestamp = target.timestamp
    assert select_patient_pairs([source, target], [admission()])[0] == []


def test_pa_is_preferred_when_both_endpoints_share_both_frontal_views():
    source, target = study(1, 0), study(2, 24)
    for item in (source, target):
        item.images_by_view["PA"] = ImageInfo(
            "pa" + item.study_id, "PA", item.timestamp, Path("/unused.jpg"), "unused.jpg"
        )
    rows, _ = select_patient_pairs([source, target], [admission()])
    assert rows[0]["matched_view"] == "PA"


@pytest.mark.parametrize("kwargs", [
    {"mode": "unknown"}, {"linkage": "unknown"}, {"seed": True},
    {"random_pairs_per_patient": -1}, {"random_pairs_per_patient": 1.5},
    {"min_gap_hours": float("nan")}, {"max_gap_days": float("inf")},
    {"min_gap_hours": 48, "max_gap_days": 1},
])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        select_patient_pairs([], [], **kwargs)
