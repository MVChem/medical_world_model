"""Independently check complete pilot scores with scikit-learn before export."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer

from .common import atomic, digest, read, rows, verify


def main(run):
    protocol = verify(run)
    if read(run / "status.json")["status"] != "complete":
        raise ValueError("All models must finish before final validation")
    references = rows(run / "references.jsonl")
    expected_ids = {r["id"] for r in references}
    if len(expected_ids) != protocol["n"]:
        raise ValueError("Reference IDs are not unique or count changed")
    assert {r["id"] for r in rows(run / "inputs.jsonl")} == expected_ids
    excluded = set(read(run / "training_overlap.json")["overlap_sample_ids"])
    vocabulary = read(run / "vocabulary.json")
    invalid = "__invalid_output__"
    mlb = MultiLabelBinarizer(classes=[*vocabulary, invalid])
    mlb.fit([[]])
    validation = {}
    for model in read(run / "models.json"):
        name = model["id"]
        raw = rows(run / name / "predictions.jsonl")
        predictions = {r["id"]: r for r in raw if r["ok"]}
        assert set(predictions) == expected_ids
        assert len(raw) == len(predictions), (
            "Unexpected retries/duplicates; review individually"
        )
        status = read(run / name / "status.json")
        assert status["status"] == "complete" and status["completed"] == len(references)
        metrics = read(run / name / "metrics.json")
        assert metrics["complete"]
        assert metrics["protocol_sha256"] == digest(run / "protocol.json")
        assert metrics["predictions_sha256"] == digest(run / name / "predictions.jsonl")
        target, predicted = [], []
        for ref in references:
            target.append(ref["answer"])
            prediction = predictions[ref["id"]]
            # The primary structured run must generate JSON directly, not a fence or prose.
            if prediction["finish_reason"] == "length":
                predicted.append([invalid])
                continue
            try:
                values = json.loads(prediction["text"])
                assert isinstance(values, list) and all(
                    isinstance(x, str) for x in values
                )
                values = [" ".join(x.lower().split()) for x in values]
                assert set(values) <= set(vocabulary)
                predicted.append(values)
            except (ValueError, TypeError, AssertionError):
                predicted.append([invalid])
        y_true, y_pred = mlb.transform(target), mlb.transform(predicted)
        masks = {
            "overall": np.ones(len(references), dtype=bool),
            "diagnosis": np.array(
                [
                    r["id"] not in excluded
                    and r["content_type"] not in ("plane", "gender")
                    for r in references
                ]
            ),
            "diagnosis_all_sampled": np.array(
                [r["content_type"] not in ("plane", "gender") for r in references]
            ),
            "vqa_train_disjoint": np.array(
                [r["id"] not in excluded for r in references]
            ),
        }
        validation[name] = {}
        for subset, mask in masks.items():
            exact = accuracy_score(y_true[mask], y_pred[mask])
            micro = f1_score(
                y_true[mask], y_pred[mask], average="micro", zero_division=0
            )
            np.testing.assert_allclose(
                [exact, micro],
                [metrics[subset]["exact_match"], metrics[subset]["micro_f1"]],
                atol=1e-12,
            )
            assert metrics[subset]["n"] == int(mask.sum())
            validation[name][subset] = {
                "n": int(mask.sum()),
                "sklearn_exact_match": exact,
                "sklearn_micro_f1": micro,
            }
    result = {
        "status": "passed",
        "models": validation,
        "validation_source_sha256": digest(__file__),
        "protocol_sha256": digest(run / "protocol.json"),
        "checks": [
            "frozen source and artifact hashes",
            "identical unique sample IDs",
            "all responses present",
            "unmodified recorded predictions",
            "fixed training-patient exclusion",
            "independent sklearn EM/micro-F1",
        ],
    }
    atomic(run / "validation.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    main(parser.parse_args().run.resolve())
