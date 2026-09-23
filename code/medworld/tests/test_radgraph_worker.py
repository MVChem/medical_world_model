"""Worker contract failures are tested without importing a model or using GPU."""
import hashlib
import io
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from medworld.evaluation.future_metrics import OfficialRadGraphScorer


def provenance():
    return {"package_version": "0.1.18", "model_type": "radgraph-xl",
            "reward_level": "partial", "report_scope": "full_report",
            "model_files_sha256": {name: hashlib.sha256(b"fixture").hexdigest()
                                    for name in ("weights.th", "config.json")},
            "scorer_files_sha256": {name: "0" * 64 for name in ("radgraph.py", "rewards.py", "utils.py")}}


def runtime():
    return {"site_packages": "runtime/site-packages", "batch_size": 8,
            "versions": {"transformers": "4.51.3", "tokenizers": "0.21.4", "huggingface-hub": "0.34.4"},
            "files_sha256": {"dependency.py": hashlib.sha256(b"fixture").hexdigest()}}


def test_missing_offline_model_or_runtime_fails_before_worker(tmp_path):
    cache = tmp_path / "models"
    tokenizer = tmp_path / "tokenizers"
    tokenizer.mkdir()
    with patch("subprocess.Popen", side_effect=AssertionError("worker must not start")):
        with pytest.raises(ValueError, match="inventory"):
            OfficialRadGraphScorer(provenance(), model_cache_dir=cache,
                                   tokenizer_cache_dir=tokenizer, worker_runtime=runtime())
        model = cache / "radgraph-xl"
        model.mkdir(parents=True)
        for name in ("weights.th", "config.json"):
            (model / name).write_bytes(b"fixture")
        with pytest.raises(ValueError, match="runtime must exist"):
            OfficialRadGraphScorer(provenance(), model_cache_dir=cache,
                                   tokenizer_cache_dir=tokenizer, worker_runtime=runtime())
        overlay = tmp_path / "runtime" / "site-packages"
        overlay.mkdir(parents=True)
        (overlay / "dependency.py").write_bytes(b"changed")
        with pytest.raises(ValueError, match="hash mismatch"):
            OfficialRadGraphScorer(provenance(), model_cache_dir=cache,
                                   tokenizer_cache_dir=tokenizer, worker_runtime=runtime())


def test_worker_transport_bounds_batches_and_retains_every_report():
    scorer = object.__new__(OfficialRadGraphScorer)
    scorer.provenance = provenance()
    scorer.worker_runtime = runtime()
    responses = [{"state": "ok", "scores": [0.5] * 8},
                 {"state": "ok", "scores": [1.0] * 8}, {"state": "ok", "scores": [0.0]}]
    outgoing = io.StringIO()
    scorer._worker = SimpleNamespace(stdin=outgoing,
        stdout=io.StringIO("\n".join(json.dumps(x) for x in responses) + "\n"), wait=Mock())
    scores = scorer.score([f"reference{i}" for i in range(17)], [f"prediction{i}" for i in range(17)])
    assert scores == [0.5] * 8 + [1.0] * 8 + [0.0]
    requests = [json.loads(line) for line in outgoing.getvalue().splitlines()]
    assert [len(row["refs"]) for row in requests] == [8, 8, 1]
    assert [value for row in requests for value in row["hyps"]] == [f"prediction{i}" for i in range(17)]
    scorer.close()


def test_worker_failure_is_not_a_numeric_or_lexical_score():
    scorer = object.__new__(OfficialRadGraphScorer)
    scorer.worker_runtime = runtime()
    scorer._worker = SimpleNamespace(stdin=io.StringIO(),
        stdout=io.StringIO('{"state":"failed","error":"missing local asset"}\n'), wait=Mock())
    with pytest.raises(RuntimeError, match="missing local asset"):
        scorer.score(["No edema"], ["No edema"])
    scorer.close()
