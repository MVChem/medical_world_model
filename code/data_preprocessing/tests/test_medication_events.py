from datetime import datetime

import pytest

from data_preprocessing.medication_events import (
    emar_event, event_overlaps, input_event, normalize_status, parse_timestamp,
)


def emar(**changes):
    return {"subject_id": "p", "hadm_id": "h", "emar_id": "p-1", "medication": "Acetaminophen",
            "event_txt": "Administered", "charttime": "2120-01-02 01:00:00", **changes}


def icu(**changes):
    return {"subject_id": "p", "hadm_id": "h", "itemid": "1", "orderid": "11",
            "starttime": "2120-01-01 20:00:00", "endtime": "2120-01-02 02:00:00",
            "statusdescription": "Stopped", "amount": "60", "amountuom": "mg",
            "rate": "10", "rateuom": "mg/hour", **changes}


ITEM = {"itemid": "1", "label": "Propofol", "linksto": "inputevents", "category": "Medications"}
LOWER, UPPER = "2120-01-02 00:00:00", "2120-01-03 00:00:00"


@pytest.mark.parametrize("status", ["Administered", "Applied", "Started", "Restarted", "Given", "  ADMINISTERED  "])
def test_emar_explicit_administration_without_four_class_restriction(status):
    event, reason = emar_event(emar(event_txt=status))
    assert reason is None
    assert event["name"] == "Acetaminophen"
    assert event["start"] == event["end"]
    assert event["amount"] is None
    assert event_overlaps(event, "p", LOWER, UPPER)


@pytest.mark.parametrize("status", ["Not Given", "Confirmed", "Held", "Stopped", "Cancelled", "", "Unknown"])
def test_emar_nonadministration_is_excluded(status):
    event, reason = emar_event(emar(event_txt=status))
    assert event is None
    assert reason.startswith("not_explicit_administration:")


def test_emar_details_exact_join_no_planned_dose_or_product_summation():
    details = [
        {"subject_id": "p", "emar_id": "p-1", "dose_due": "1000", "route": "PO"},
        {"subject_id": "p", "emar_id": "p-1", "dose_given": "250", "dose_given_unit": "mg", "route": "PO"},
        {"subject_id": "p", "emar_id": "p-1", "dose_given": "250", "dose_given_unit": "mg", "route": "PO"},
        {"subject_id": "other", "emar_id": "p-1", "dose_given": "0"},
        {"subject_id": "p", "emar_id": "other", "dose_given": "0"},
    ]
    event, _ = emar_event(emar(), details)
    assert event["amount"] is None
    assert event["route"] == "PO"
    assert len(event["dose_details"]) == 3
    event, _ = emar_event(emar(), details[:1])
    assert event["amount"] is None
    event, _ = emar_event(emar(), details[1:2])
    assert (event["amount"], event["amount_unit"]) == ("250", "mg")


@pytest.mark.parametrize("values,rejected", [(["0"], True), ([0], True), (["-1", "0"], True),
                                           (["0", "1"], False), ([""], False),
                                           (["0", "Unknown"], False), (["NaN"], False)])
def test_emar_nonpositive_dose_evidence(values, rejected):
    details = [{"subject_id": "p", "emar_id": "p-1", "dose_given": value} for value in values]
    event, reason = emar_event(emar(), details)
    assert (event is None) is rejected
    assert reason == ("explicit_nonpositive_dose" if rejected else None)


@pytest.mark.parametrize("status", ["FinishedRunning", "ChangeDose/Rate", "ChangedDose/Rate", "Changed", "Stopped", "Paused", "Finished Running"])
def test_icu_positive_delivered_medication_segments(status):
    event, reason = input_event(icu(statusdescription=status), ITEM)
    assert reason is None
    assert event["event_type"] == "infusion_segment"
    assert event["name"] == "Propofol"
    assert event_overlaps(event, "p", LOWER, UPPER)
    assert (event["amount"], event["amount_unit"], event["rate_unit"]) == ("60", "mg", "mg/hour")
    assert event["start"] < parse_timestamp(LOWER)  # overlapping segment began before the source CXR


@pytest.mark.parametrize("status", ["Rewritten", "Cancelled", "Flushed", "Unknown", ""])
def test_icu_rewrites_and_undelivered_states_are_excluded(status):
    event, reason = input_event(icu(statusdescription=status), ITEM)
    assert event is None
    assert reason.startswith("not_delivered_segment_status:")


@pytest.mark.parametrize("amount", ["", "0", "-1", "NaN", "inf", None])
def test_icu_requires_finite_positive_delivered_amount(amount):
    assert input_event(icu(amount=amount), ITEM) == (None, "missing_or_nonpositive_amount")


@pytest.mark.parametrize("category", ["Fluids/Intake", "Blood Products/Colloids", "Fluids - Other (Not In Use)",
                                      "Nutrition - Enteral", "Nutrition - Parenteral", "Nutrition - Supplements", ""])
def test_icu_nonmedication_categories_are_excluded(category):
    assert input_event(icu(), ITEM | {"category": category}) == (None, "not_medication_item_category")


def test_dictionary_must_match_item_and_original_input_table():
    for item in [None, ITEM | {"itemid": "2"}, ITEM | {"linksto": "ingredientevents"}]:
        assert input_event(icu(), item) == (None, "missing_or_wrong_item_dictionary")
    event, _ = input_event(icu(), ITEM | {"category": "Antibiotics", "label": "Cefazolin"})
    assert event["name"] == "Cefazolin"


@pytest.mark.parametrize("rate", ["", "0", "-1", "NaN", "inf", None])
def test_no_rate_does_not_assume_continuous_exposure(rate):
    event, _ = input_event(icu(rate=rate), ITEM)
    assert event["event_type"] == "administration"
    assert event["start"] == event["end"] < event["source_end"]
    assert not event_overlaps(event, "p", LOWER, UPPER)
    event, _ = input_event(icu(rate=rate, starttime=LOWER), ITEM)
    assert event_overlaps(event, "p", LOWER, UPPER)


def test_overlap_requires_same_patient_without_expanded_web_window():
    for time, retained in [(LOWER, True), (UPPER, True), ("2120-01-01 23:59:59", False),
                           ("2120-01-03 00:00:01", False)]:
        event, _ = emar_event(emar(charttime=time))
        assert event_overlaps(event, "p", LOWER, UPPER) is retained
    event, _ = emar_event(emar())
    assert not event_overlaps(event, "p", LOWER, UPPER, hadm_id="other")
    assert not event_overlaps(event, "other", LOWER, UPPER)
    assert not event_overlaps(event, "p", LOWER, UPPER, hadm_id="")
    with pytest.raises(ValueError, match="ordered naive"):
        event_overlaps(event, "p", UPPER, LOWER)


@pytest.mark.parametrize("admission", [None, "", "other"])
def test_patient_only_scope_accepts_missing_or_different_admissions(admission):
    for event, reason in [emar_event(emar(hadm_id=admission)), input_event(icu(hadm_id=admission), ITEM)]:
        assert reason is None
        assert event["hadm_id"] == (admission or "")
        assert event_overlaps(event, "p", LOWER, UPPER)
        assert not event_overlaps(event, "p", LOWER, UPPER, hadm_id="h")
        assert not event_overlaps(event, "other", LOWER, UPPER)


def test_patient_only_scope_has_no_fixed_gap_limit():
    event, _ = emar_event(emar())
    assert event_overlaps(event, "p", "2110-01-01 00:00:00", "2130-01-01 00:00:00")
    assert event_overlaps(event, "p", "2120-01-02 00:59:59", "2120-01-02 01:00:01")


@pytest.mark.parametrize("bad", [None, "", "not-time", "2120-01-02", "2120-01-02 01:00:00+00:00"])
def test_invalid_or_non_naive_timestamp_is_rejected(bad):
    assert parse_timestamp(bad) is None
    assert emar_event(emar(charttime=bad)) == (None, "missing_or_invalid_event_time")
    assert input_event(icu(endtime=bad), ITEM) == (None, "missing_or_invalid_event_time")


def test_reverse_segment_empty_identity_and_stable_row_provenance():
    assert input_event(icu(endtime="2120-01-01 19:00:00"), ITEM)[1] == "missing_or_invalid_event_time"
    assert emar_event(emar(subject_id=""))[1] == "missing_subject"
    assert input_event(icu(subject_id=""), ITEM)[1] == "missing_subject"
    assert emar_event(emar(medication=" "))[1] == "missing_medication_name"
    event, _ = emar_event(emar())
    reordered, _ = emar_event(dict(reversed(list(emar().items()))))
    changed, _ = emar_event(emar(charttime=LOWER))
    assert event["record_id"] == reordered["record_id"] != changed["record_id"]
    assert event["record_id"].startswith("hosp.emar:p-1:")
    assert parse_timestamp(datetime(2120, 1, 2)) == datetime(2120, 1, 2)
    assert normalize_status(" NOT   Given ") == "not given"
