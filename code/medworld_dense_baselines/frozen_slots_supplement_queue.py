"""Run additional shuffled controls against an immutable parent experiment.

Use the original experiment's frozen training source and caches, a fresh run
directory, and explicit --controls. The underlying queue retains GPU occupancy
checks, per-card locks, durable workers, and checkpoint recovery.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import frozen_slots_queue as queue


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    known, _ = parser.parse_known_args()
    run = known.run.resolve()
    protocol = json.loads((run / "protocol.json").read_text())
    parent = Path(protocol["parent_run"]).resolve()
    reference = json.loads((run / "parent_reference.json").read_text())
    if reference["parent_run"] != str(parent):
        raise ValueError("Parent reference does not match the supplement protocol")
    for relative, expected in reference["artifacts"].items():
        if queue.digest(parent / relative) != expected:
            raise ValueError(f"Parent artifact changed: {relative}")
    parent_settings = json.loads((parent / "queue_config.json").read_text())["settings"]
    for name, expected in parent_settings["source_sha256"].items():
        if queue.digest(known.source / name) != expected:
            raise ValueError(f"Supplement must preserve the frozen parent source: {name}")
    original_make_jobs = queue.make_jobs

    def make_jobs(run, models, controls, epochs, seed, batch_size, microbatch):
        requested = dict(epochs=epochs, seed=seed, batch_size=batch_size, microbatch=microbatch)
        if any(value != parent_settings[key] for key, value in requested.items()):
            raise ValueError("Supplement training budget must match the parent baseline")
        if set(controls) != set(protocol["supplementary_shuffled_models"]):
            raise ValueError("Requested controls differ from the supplement protocol")
        jobs = original_make_jobs(run, models, controls, epochs, seed, batch_size, microbatch)
        jobs = [job for job in jobs if job["id"].endswith("_shuffled_slots")]
        for job in jobs:
            job["deps"] = []
            # This exact path is part of the matched baseline's contract.
            job["args"][job["args"].index("--data-run") + 1] = str(parent)
        return jobs

    def write_report(run, source):
        from frozen_slots_supplement_report import report
        try:
            report(run, source)
        except Exception:
            import traceback
            traceback.print_exc()
            # Training progress remains durable even if reporting fails.

    queue.make_jobs = make_jobs
    queue.write_report = write_report
    return queue.main()


if __name__ == "__main__":
    sys.exit(main())
