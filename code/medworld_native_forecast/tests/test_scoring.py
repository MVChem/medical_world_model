"""Scoring DAG, bridge cohort integrity, and incomplete-metric handling."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def module(name):
    spec = importlib.util.spec_from_file_location("native_test_" + name, ROOT / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


queue = module("launch_queue")
scoring = module("score_results")
green = module("green_bridge")


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def cohort(root):
    cache = root / "cache"
    pairs = [dict(id=f"pair{i}", source=f"source{i}", target=f"target{i}") for i in range(297)]
    write_rows(cache / "test.jsonl", pairs)
    write_rows(cache / "observations.jsonl", [dict(id=f"target{i}", report=f"reference {i}") for i in range(297)])
    for condition in green.CONDITIONS:
        out = root / condition / "evaluation_test"
        write_rows(out / "predictions.jsonl", [dict(id=row["id"], report="prediction", scores=[0.5]) for row in pairs])
        (out / "generation.json").write_text(json.dumps(dict(state="complete", smoke_only=False,
            split="test", condition=condition, count=297)))
        (out / "config.json").write_text(json.dumps(dict(cache=str(cache))))
    return pairs


def test_queue_score_implies_evaluate_and_orders_all_stages(tmp_path):
    jobs = queue.jobs_for(tmp_path, tmp_path / "config.json", False, False, True)
    assert len(jobs) == 11
    for condition in ("native", "slots", "shuffled"):
        assert jobs[condition]["depends"] == ["stage1"]
        assert jobs[condition + "_test"]["depends"] == [condition]
        assert jobs[condition + "_clinical"]["depends"] == [condition + "_test"]
    assert set(jobs["green"]["depends"]) == {name + "_clinical" for name in ("native", "slots", "shuffled")}
    with pytest.raises(ValueError, match="Smoke"):
        queue.jobs_for(tmp_path, tmp_path / "config.json", True, False, True)
    assert 5 not in queue.ALLOWED_GPUS and 6 not in queue.ALLOWED_GPUS


def test_scoring_adds_only_needed_legacy_config_fields(tmp_path):
    cfg = dict(qwen="new_model", cache="new_cohort", retrieval_negatives=31)
    fallback = tmp_path / "old.json"
    fallback.write_text(json.dumps(dict(qwen="old_model", data_root="raw", vjepa_checkpoint="jepa")))
    result = scoring.scoring_config(cfg, fallback)
    assert result == dict(qwen="new_model", cache="new_cohort", retrieval_negatives=3,
                          data_root="raw", vjepa_checkpoint="jepa")
    assert cfg["retrieval_negatives"] == 31 and "data_root" not in cfg


def test_green_bridge_requires_exact_297_and_reports_missing_metrics(tmp_path):
    pairs = cohort(tmp_path)
    bridge, expected = green.build_bridge(tmp_path)
    assert expected == [row["id"] for row in pairs]
    references = scoring.rows(bridge / "cohort/table1_references_test.jsonl")
    assert len(references) == 297 and references[0]["target_report"] == "reference 0"
    for condition in green.CONDITIONS:
        responses = scoring.rows(bridge / condition / "test/responses.jsonl")
        assert responses[0] == dict(key="table1_report|pair0|", ok=True, text="prediction")
    assert green.validate_green(tmp_path, bridge, expected) is False
    status = json.loads((bridge / "bridge_status.json").read_text())
    assert status["status"] == "unavailable" and status["table1_ready"] is False
    green.build_bridge(tmp_path, resume=True)
    path = tmp_path / "native/evaluation_test/predictions.jsonl"
    predicted = scoring.rows(path)
    predicted[0]["id"] = "wrong-id"
    write_rows(path, predicted)
    with pytest.raises(ValueError, match="IDs/order"):
        green.build_bridge(tmp_path, resume=True)


def test_green_completion_checks_each_condition_and_response_ids(tmp_path):
    cohort(tmp_path)
    bridge, expected = green.build_bridge(tmp_path)
    for condition in green.CONDITIONS:
        directory = bridge / condition / "test"
        (directory / "green_metrics.json").write_text(json.dumps(dict(status="complete", n=297,
            completed=297, mean=0.5, invalid_outputs=0)))
        write_rows(directory / "green_responses.jsonl", [dict(id=identity, score=0.5, valid=True) for identity in expected])
    assert green.validate_green(tmp_path, bridge, expected)
    write_rows(bridge / "slots/test/green_responses.jsonl", [dict(id=identity, score=0.5) for identity in expected[:-1]])
    assert not green.validate_green(tmp_path, bridge, expected)
