"""Strict Table 1 scorers; these do not supply forecasting heads or references.

Every reference is ``{"id": str, "patient": str, "target": value}``;
progression references additionally contain ``finding``. Predictions map those
IDs to values. References must already contain only independently assessed,
eligible targets, frozen before inference. Missing predictions never change
the denominator. The caller must enforce source-only model inputs.

Protocol keys shared by all tasks are ``schema``, ``task``, ``unit``, and
``reference_sha256``. VQA adds ``vocabulary``; progression adds ``findings``;
mortality adds ``horizon_days=30``; reports add ``radgraph`` provenance.
Scores remain fractions except LOS, which is in days.
"""

import hashlib
import importlib.metadata
import json
import math
import numbers
import os
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path


SCHEMA = "medworld-future-metrics-v1"
DIRECTIONS = ("improved", "stable", "worsened")
TASKS = ("future_vqa", "progression", "future_report", "mortality_30d", "remaining_los")
RADGRAPH_PACKAGE_VERSION = "0.1.18"


def _hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def reference_fingerprint(references):
    """Order-independent hash of the complete fixed reference records."""
    return _hash(sorted(references, key=lambda row: row["id"]))


def _text(value):
    return " ".join(value.casefold().split()) if isinstance(value, str) else None


def _number(value, maximum=None):
    return (isinstance(value, numbers.Real) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0
            and (maximum is None or value <= maximum))


def _names(values, name, *, normalize=False):
    if (not isinstance(values, list) or not values
            or any(not isinstance(x, str) or not x.strip() for x in values)):
        raise ValueError(f"{name} must be a nonempty string list")
    values = [_text(x) for x in values] if normalize else values
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicates")
    return values


def _validate(task, references, predictions, protocol):
    if task not in TASKS:
        raise ValueError(f"Unknown future task: {task}")
    extra = {"future_vqa": {"vocabulary"}, "progression": {"findings"},
             "future_report": {"radgraph"}, "mortality_30d": {"horizon_days"},
             "remaining_los": set()}[task]
    expected = {"schema", "task", "unit", "reference_sha256"} | extra
    if not isinstance(protocol, Mapping) or set(protocol) != expected:
        raise ValueError(f"Protocol keys for {task} must be {sorted(expected)}")
    unit = "days" if task == "remaining_los" else "fraction"
    if protocol["schema"] != SCHEMA or protocol["task"] != task or protocol["unit"] != unit:
        raise ValueError("Future metric schema, task or unit mismatch")
    if task == "mortality_30d" and (type(protocol["horizon_days"]) is not int
                                   or protocol["horizon_days"] != 30):
        raise ValueError("Mortality horizon_days must be exactly 30")
    if not isinstance(references, (list, tuple)) or not references:
        raise ValueError("A nonempty frozen reference cohort is required")
    if not isinstance(predictions, Mapping):
        raise ValueError("Predictions must map reference IDs to values")
    keys = {"id", "patient", "target"} | ({"finding"} if task == "progression" else set())
    ids = []
    for row in references:
        if not isinstance(row, Mapping) or set(row) != keys:
            raise ValueError(f"Reference keys for {task} must be {sorted(keys)}")
        if any(not isinstance(row[key], str) or not row[key].strip() for key in ("id", "patient")):
            raise ValueError("Reference ID and patient must be nonempty strings")
        ids.append(row["id"])
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate reference IDs")
    if set(predictions) - set(ids):
        raise ValueError("Predictions contain IDs outside the frozen cohort")
    fingerprint = reference_fingerprint(references)
    if protocol["reference_sha256"] != fingerprint:
        raise ValueError("Frozen reference cohort hash mismatch")
    missing = [r["id"] for r in references if r["id"] not in predictions]
    return {"schema": SCHEMA, "task": task, "unit": unit, "complete": not missing,
            "n": len(references), "patients": len({r["patient"] for r in references}),
            "reference_sha256": fingerprint, "protocol_sha256": _hash(protocol),
            "missing_ids": missing}


def score_future_task(task, references, predictions, *, protocol, radgraph_scorer=None):
    """Score the entire pinned cohort, or return an explicit incomplete result.

    Malformed categorical answers are wrong answers. Invalid numeric scores
    make the aggregate unavailable rather than silently evaluating a subset.
    Unsupported reference classes likewise leave the metric unavailable.
    """
    result = _validate(task, references, predictions, protocol)
    if task == "future_vqa":
        vocabulary = set(_names(protocol["vocabulary"], "vocabulary", normalize=True))
        support, correct, invalid = Counter(), 0, []
        for row in references:
            target = _text(row["target"])
            if target not in vocabulary:
                raise ValueError(f"Invalid VQA reference: {row['id']}")
            prediction = _text(predictions.get(row["id"]))
            support[target] += 1
            if prediction not in vocabulary:
                invalid.append(row["id"])
            else:
                correct += prediction == target
        return {**result, "accuracy": correct / result["n"] if result["complete"] else None,
                "correct": correct,
                "invalid": len(invalid), "invalid_ids": invalid,
                "support": {name: support[name] for name in sorted(vocabulary)}}

    if task == "progression":
        findings = _names(protocol["findings"], "findings")
        per_finding = {finding: {"support": Counter(), "correct": Counter(), "invalid": 0}
                       for finding in findings}
        invalid = []
        total_support, total_correct = Counter(), Counter()
        for row in references:
            finding, target = row["finding"], row["target"]
            if not isinstance(finding, str) or finding not in per_finding or target not in DIRECTIONS:
                raise ValueError(f"Invalid assessable progression reference: {row['id']}")
            prediction = _text(predictions.get(row["id"]))
            group = per_finding[finding]
            group["support"][target] += 1
            total_support[target] += 1
            if prediction not in DIRECTIONS:
                group["invalid"] += 1
                invalid.append(row["id"])
            else:
                group["correct"][target] += prediction == target
                total_correct[target] += prediction == target
        unsupported = []
        for finding, group in per_finding.items():
            support = {direction: group["support"][direction] for direction in DIRECTIONS}
            recalls = {direction: group["correct"][direction] / count if count else None
                       for direction, count in support.items()}
            score = sum(recalls.values()) / len(DIRECTIONS) if all(support.values()) else None
            if score is None:
                unsupported.append(finding)
            per_finding[finding] = {"n": sum(support.values()), "support": support,
                                    "recall": recalls, "balanced_accuracy": score,
                                    "invalid": group["invalid"]}
        support = {direction: total_support[direction] for direction in DIRECTIONS}
        recalls = {direction: total_correct[direction] / count if count else None
                   for direction, count in support.items()}
        unsupported_directions = [direction for direction in DIRECTIONS if not support[direction]]
        complete = result["complete"] and not unsupported_directions
        return {**result, "complete": complete,
                "balanced_accuracy": sum(recalls.values()) / len(DIRECTIONS) if complete else None,
                "invalid": len(invalid), "invalid_ids": invalid,
                "support": support, "recall": recalls,
                "directions": list(DIRECTIONS), "per_finding": per_finding,
                "unsupported_findings": unsupported, "unsupported_directions": unsupported_directions}

    if task == "future_report":
        provenance = validate_radgraph_provenance(protocol["radgraph"])
        if not isinstance(radgraph_scorer, OfficialRadGraphScorer):
            raise ValueError("Future reports require an official RadGraph scorer; no lexical fallback")
        if radgraph_scorer.provenance != provenance:
            raise ValueError("RadGraph scorer provenance mismatch")
        refs, hyps, invalid = [], [], []
        for row in references:
            if not isinstance(row["target"], str) or not row["target"].strip():
                raise ValueError(f"Invalid reference report: {row['id']}")
            prediction = predictions.get(row["id"])
            if not isinstance(prediction, str) or not prediction.strip():
                invalid.append(row["id"])
                prediction = ""
            refs.append(row["target"])
            hyps.append(prediction)
        try:
            scores = radgraph_scorer.score(refs, hyps)
        except Exception as exc:
            return {**result, "complete": False, "radgraph_f1": None,
                    "invalid": len(invalid), "invalid_ids": invalid,
                    "failure": f"Official RadGraph scoring failed: {type(exc).__name__}: {exc}",
                    "radgraph": provenance}
        return {**result, "radgraph_f1": sum(scores) / result["n"] if result["complete"] else None,
                "per_report": [{"id": row["id"], "f1": score}
                               for row, score in zip(references, scores)],
                "invalid": len(invalid), "invalid_ids": invalid, "radgraph": provenance}

    maximum = 1 if task == "mortality_30d" else None
    invalid, targets, scores = [], [], []
    for row in references:
        target = row["target"]
        if not _number(target, maximum) or (task == "mortality_30d" and target not in (0, 1)):
            raise ValueError(f"Invalid {task} reference: {row['id']}")
        targets.append(target)
        score = predictions.get(row["id"])
        if not _number(score, maximum):
            invalid.append(row["id"])
        scores.append(score)
    key = "auroc" if task == "mortality_30d" else "mae_days"
    result.update({"invalid": len(invalid), "invalid_ids": invalid})
    if task == "mortality_30d":
        result["support"] = {"negative": targets.count(0), "positive": targets.count(1)}
    if invalid:
        return {**result, "complete": False, key: None,
                "failure": "Missing, nonfinite, or out-of-range numeric predictions"}
    if task == "mortality_30d":
        if not all(result["support"].values()):
            return {**result, "complete": False, key: None,
                    "failure": "AUROC requires both reference classes"}
        from sklearn.metrics import roc_auc_score
        value = float(roc_auc_score(targets, scores))
    else:
        value = sum(abs(score - target) for score, target in zip(scores, targets)) / result["n"]
    return {**result, key: value}


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_radgraph_provenance(provenance):
    """Pin the official XL partial variant and local scorer/model artifacts."""
    keys = {"package_version", "model_type", "reward_level", "report_scope",
            "model_files_sha256", "scorer_files_sha256"}
    if not isinstance(provenance, dict) or set(provenance) != keys:
        raise ValueError(f"RadGraph provenance keys must be {sorted(keys)}")
    if (provenance["package_version"] != RADGRAPH_PACKAGE_VERSION
            or provenance["model_type"] != "radgraph-xl"
            or provenance["reward_level"] != "partial"
            or provenance["report_scope"] != "full_report"):
        raise ValueError("RadGraph requires version 0.1.18, XL, partial F1, full-report scope")
    for field in ("model_files_sha256", "scorer_files_sha256"):
        files = provenance[field]
        if not isinstance(files, dict) or not files:
            raise ValueError(f"{field} requires pinned file hashes")
        for name, digest in files.items():
            if (not isinstance(name, str) or not name or Path(name).is_absolute()
                    or ".." in Path(name).parts or not isinstance(digest, str)
                    or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
                raise ValueError(f"Invalid artifact path/hash in {field}")
    if not {"weights.th", "config.json"} <= set(provenance["model_files_sha256"]):
        raise ValueError("RadGraph model hashes must include weights.th and config.json")
    if not {"radgraph.py", "rewards.py", "utils.py"} <= set(provenance["scorer_files_sha256"]):
        raise ValueError("RadGraph scorer hashes must include radgraph.py, rewards.py and utils.py")
    return json.loads(json.dumps(provenance))


def _verify_files(root, expected, *, exact=False):
    root = Path(root).resolve()
    if exact and {p.relative_to(root).as_posix() for p in root.rglob("*")
                  if p.is_file() and p.name != ".lock" and "__pycache__" not in p.parts
                  and p.suffix != ".pyc"} != set(expected):
        raise ValueError("RadGraph local model inventory differs from pinned files")
    for name, digest in expected.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or _sha256_file(path) != digest:
            raise ValueError(f"RadGraph local artifact missing or hash mismatch: {name}")


class OfficialRadGraphScorer:
    """Explicit, lazy, offline loader for the official F1RadGraph adapter.

    Launch a fresh evaluation process with HF_HUB_OFFLINE=1 and
    TRANSFORMERS_OFFLINE=1, and stage/hash model and tokenizer assets first.
    No network fallback or report-string similarity approximation is supplied.
    Official API: https://github.com/Stanford-AIMI/radgraph
    """

    def __init__(self, provenance, *, model_cache_dir, tokenizer_cache_dir, cuda=-1, worker_runtime=None):
        self.provenance = validate_radgraph_provenance(provenance)
        if worker_runtime is None and any(os.environ.get(key) != "1" for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")):
            raise ValueError("Set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 before launching RadGraph")
        _verify_files(Path(model_cache_dir) / "radgraph-xl", self.provenance["model_files_sha256"], exact=True)
        if not Path(tokenizer_cache_dir).is_dir():
            raise ValueError("RadGraph requires a prepopulated local tokenizer cache")
        self._worker = None
        if worker_runtime is not None:
            self._start_worker(worker_runtime, model_cache_dir, tokenizer_cache_dir)
            return
        if importlib.metadata.version("radgraph") != RADGRAPH_PACKAGE_VERSION:
            raise ValueError("Installed RadGraph package does not match pinned version")
        import huggingface_hub.constants
        import transformers.utils.hub
        if not huggingface_hub.constants.HF_HUB_OFFLINE or not transformers.utils.hub.is_offline_mode():
            raise ValueError("RadGraph offline flags must be active before library imports")
        import radgraph
        _verify_files(Path(radgraph.__file__).parent, self.provenance["scorer_files_sha256"])
        from radgraph import F1RadGraph
        self._scorer = F1RadGraph(reward_level="partial", model_type="radgraph-xl",
                                 model_cache_dir=str(model_cache_dir),
                                 tokenizer_cache_dir=str(tokenizer_cache_dir), cuda=cuda)

    def _start_worker(self, runtime, model_cache_dir, tokenizer_cache_dir):
        keys = {"site_packages", "versions", "files_sha256", "batch_size"}
        versions = {"transformers": "4.51.3", "tokenizers": "0.21.4", "huggingface-hub": "0.34.4"}
        if (not isinstance(runtime, dict) or set(runtime) != keys or runtime["versions"] != versions
                or not isinstance(runtime["site_packages"], str) or not runtime["site_packages"].strip()
                or not isinstance(runtime["files_sha256"], dict) or not runtime["files_sha256"]
                or type(runtime["batch_size"]) is not int or not 1 <= runtime["batch_size"] <= 8):
            raise ValueError("RadGraph worker requires the pinned isolated runtime and batch_size 1..8")
        root = Path(model_cache_dir).resolve().parent
        overlay = (root / runtime["site_packages"]).resolve()
        if not overlay.is_relative_to(root) or not overlay.is_dir():
            raise ValueError("RadGraph isolated runtime must exist inside its asset directory")
        _verify_files(overlay, runtime["files_sha256"], exact=True)
        environment = dict(os.environ)
        code_root = Path(__file__).resolve().parents[2]
        environment.update(PYTHONPATH=os.pathsep.join([str(overlay), str(code_root)]),
                           CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                           HF_HUB_CACHE=str(tokenizer_cache_dir), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
        self._worker = subprocess.Popen([sys.executable, "-u", "-m", "medworld.evaluation.radgraph_worker"],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                        encoding="utf-8", env=environment)
        self.worker_runtime = json.loads(json.dumps(runtime))
        try:
            self._worker.stdin.write(json.dumps({"provenance": self.provenance,
                "model_cache_dir": str(model_cache_dir), "tokenizer_cache_dir": str(tokenizer_cache_dir),
                "versions": versions}) + "\n")
            self._worker.stdin.flush()
            ready = json.loads(self._worker.stdout.readline())
            if ready != {"state": "ready", "versions": versions}:
                raise RuntimeError(f"Official RadGraph worker failed to initialize: {ready}")
        except Exception:
            self.close()
            raise

    def close(self):
        worker = getattr(self, "_worker", None)
        if worker is None:
            return
        if worker.stdin is not None:
            worker.stdin.close()
        try:
            worker.wait(timeout=10)
        except subprocess.TimeoutExpired:
            worker.terminate()
            worker.wait(timeout=10)
        if worker.stdout is not None:
            worker.stdout.close()
        self._worker = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def score(self, refs, hyps):
        if len(refs) != len(hyps) or not refs:
            raise ValueError("RadGraph requires matched nonempty report lists")
        if getattr(self, "_worker", None) is not None:
            result = []
            size = self.worker_runtime["batch_size"]
            for start in range(0, len(refs), size):
                self._worker.stdin.write(json.dumps({"refs": refs[start:start + size],
                                                     "hyps": hyps[start:start + size]}) + "\n")
                self._worker.stdin.flush()
                response = json.loads(self._worker.stdout.readline())
                values = response.get("scores")
                if (response.get("state") != "ok" or not isinstance(values, list)
                        or len(values) != len(refs[start:start + size])
                        or any(not _number(value, 1) for value in values)):
                    raise RuntimeError(f"Official RadGraph worker scoring failed: {response}")
                result.extend(values)
            return result
        # The official implementation assigns zero to empty hypotheses. Avoid
        # invoking its extractor on an entirely empty batch, retaining all IDs.
        nonempty = [index for index, hyp in enumerate(hyps) if hyp]
        scores = [0.0] * len(refs)
        if nonempty:
            output = self._scorer(refs=[refs[i] for i in nonempty], hyps=[hyps[i] for i in nonempty])
            if not isinstance(output, (list, tuple)) or len(output) != 4:
                raise ValueError("Unexpected official RadGraph output schema")
            values = output[1]
            if len(values) != len(nonempty) or any(not _number(value, 1) for value in values):
                raise ValueError("Invalid official RadGraph per-report F1 scores")
            for index, value in zip(nonempty, values):
                scores[index] = float(value)
        return scores
