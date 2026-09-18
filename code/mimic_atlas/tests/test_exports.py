"""Export bundles retain machine artifacts and open cases in the live workspace."""

import json

import pytest

from mimic_atlas.exports import ExportLibrary


def test_export_library_streams_both_formats_and_preserves_files(tmp_path):
    root = tmp_path / "exports"
    packet = {
        "patient_id": "p123",
        "transition_id": "pair-a",
        "split": "test",
        "current_state": {"study_id": "s1"},
        "future_state": {"study_id": "s2"},
    }
    for name, row in [
        ("cxr", packet),
        (
            "linked",
            {
                "subject_id": "123",
                "mimic_cxr_transition": packet,
                "linkage_status": "unmatched",
            },
        ),
    ]:
        folder = root / "batch_20260918" / name
        folder.mkdir(parents=True)
        (folder / "summary.json").write_text(json.dumps({"generated_examples": 1}))
        (
            folder
            / ("transitions.jsonl" if name == "cxr" else "linked_transitions.jsonl")
        ).write_text(json.dumps(row) + "\n")
    library = ExportLibrary(root)
    assert [b["cases"] for b in library.catalog()] == [1, 1]
    for name in ("cxr", "linked"):
        result = library.cases("batch_20260918~" + name)
        assert result["rows"][0]["subject_id"] == "123"
        assert result["rows"][0]["transition_id"] == "pair-a"
        assert library.cases("batch_20260918~" + name, page=2)["rows"] == []
    path = library.artifact("batch_20260918~cxr", "transitions.jsonl")
    assert json.loads(path.read_text()) == packet
    for invalid in ("..", "../exports", "batch_20260918~cxr~more"):
        with pytest.raises(KeyError):
            library.folder(invalid)
    with pytest.raises(KeyError):
        library.artifact("batch_20260918~cxr", "../summary.json")
    external = tmp_path / "secret.json"
    external.write_text("private")
    (root / "batch_20260918/cxr/forecast_manifest.jsonl").symlink_to(external)
    with pytest.raises(KeyError):
        library.artifact("batch_20260918~cxr", "forecast_manifest.jsonl")
