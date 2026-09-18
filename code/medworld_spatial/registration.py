"""Brief experiment index; all detailed artifacts stay in the run directory."""
from datetime import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import tempfile
from zoneinfo import ZoneInfo

from medworld.config import PROJECT
from medworld.runtime import atomic_json


def record(run, state, *, project=PROJECT):
    run, project = Path(run).resolve(), Path(project).resolve()
    plan = json.loads((run / "plan.json").read_text())
    if not plan.get("register_experiment", False):
        return
    relative = run.relative_to(project).as_posix()
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    terminal = state in ("complete", "failed", "interrupted", "finished_with_incomplete_jobs")
    index = project / "experiments"
    index.mkdir(exist_ok=True)
    lock_id = hashlib.sha256(str(project).encode()).hexdigest()[:16]
    with (Path(tempfile.gettempdir()) / f"medworld-registry-{lock_id}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        registry = index / "registry.json"
        entries = json.loads(registry.read_text()) if registry.exists() else []
        entries = [entry for entry in entries if entry["id"] != run.name]
        if terminal:
            history = index / "README.md"
            text = history.read_text() if history.exists() else (
                "# Experiment history\n\nOngoing experiments: [registry.json](registry.json).\n\n"
                "| Date | ID | Outcome | Run |\n|---|---|---|---|\n")
            marker = "|---|---|---|---|\n"
            if marker not in text:
                raise ValueError("Experiment history table header missing")
            if f"`{run.name}`" not in text:
                outcome = (f"Completed: six variants, {plan['steps']:,} updates each; evaluation complete."
                           if state == "complete" else state.replace("_", " ").capitalize() + ".")
                row = f"| {now.date()} | `{run.name}` | {outcome} | [Run and results](../{relative}/) |\n"
                temporary = history.with_suffix(".tmp")
                temporary.write_text(text.replace(marker, marker + row, 1))
                temporary.replace(history)
        else:
            geometry = plan.get("sr_geometry", {"input_hw": [128, 128], "target_hw": [512, 512],
                                               "scale_per_axis": 4, "pixel_count_ratio": 16})
            source_size = "x".join(map(str, geometry["input_hw"]))
            target_size = "x".join(map(str, geometry["target_hw"]))
            entries.append({"id": run.name,
                "summary": f"Six matched variants; tasks: {', '.join(plan['tasks'])}; SR {source_size} to {target_size} ({geometry['scale_per_axis']}x per axis, {geometry['pixel_count_ratio']}x pixels); seed(s) {', '.join(map(str, plan['seeds']))}.",
                "status": state, "status_observed_at": now.isoformat(timespec="seconds"),
                "run_dir": relative, "live_status": relative + "/status.json"})
        atomic_json(registry, entries)
