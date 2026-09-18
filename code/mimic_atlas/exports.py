"""Discover explicit export bundles under runs/exports; never build a data cache."""

from __future__ import annotations

import json
import re
from pathlib import Path

EXPORT_FILES = {
    "summary.json",
    "transitions.jsonl",
    "linked_transitions.jsonl",
    "forecast_manifest.jsonl",
    "forecast_inputs.jsonl",
    "forecast_targets.jsonl",
    "appendix_cases.csv",
    "index.html",
}


class ExportLibrary:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def folder(self, identifier):
        if not re.fullmatch(r"[A-Za-z0-9_-]+(?:~[A-Za-z0-9_-]+)?", identifier):
            raise KeyError(identifier)
        path = self.root.joinpath(*identifier.split("~")).resolve()
        if not path.is_relative_to(self.root) or not (path / "summary.json").is_file():
            raise KeyError(identifier)
        return path

    def catalog(self):
        output = []
        for path in sorted(
            [*self.root.glob("*/summary.json"), *self.root.glob("*/*/summary.json")]
        ):
            identifier = "~".join(path.parent.relative_to(self.root).parts)
            try:
                folder = self.folder(identifier)
                summary = json.loads(path.read_text())
                output.append(
                    {
                        "id": identifier,
                        "name": folder.name,
                        "collection": str(folder.relative_to(self.root).parent),
                        "cases": summary.get("generated_examples", 0),
                        "linked": (folder / "linked_transitions.jsonl").is_file(),
                        "files": [
                            {"name": f.name, "bytes": f.stat().st_size}
                            for f in sorted(folder.iterdir())
                            if f.name in EXPORT_FILES
                            and f.is_file()
                            and not f.is_symlink()
                        ],
                    }
                )
            except (KeyError, OSError, ValueError) as exc:
                output.append(
                    {"id": identifier, "name": path.parent.name, "error": str(exc)}
                )
        return output

    def cases(self, identifier, page=1, limit=25):
        folder = self.folder(identifier)
        path = folder / (
            "linked_transitions.jsonl"
            if (folder / "linked_transitions.jsonl").is_file()
            else "transitions.jsonl"
        )
        if path.is_symlink() or not path.is_file():
            raise KeyError(identifier)
        rows, total = [], 0
        start = (page - 1) * limit
        # Stream; only this page's small identifying summaries survive the request.
        with path.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                if start <= total < start + limit:
                    row = json.loads(line)
                    packet = row.get("mimic_cxr_transition", row)
                    rows.append(
                        {
                            "subject_id": row.get("subject_id")
                            or packet["patient_id"].removeprefix("p"),
                            "transition_id": packet["transition_id"],
                            "split": packet.get("split"),
                            "source_study": packet["current_state"]["study_id"],
                            "target_study": packet["future_state"]["study_id"],
                            "category": packet.get("curation", {}).get("category", ""),
                            "linkage_status": row.get("linkage_status"),
                        }
                    )
                total += 1
        return {"rows": rows, "total": total, "page": page, "limit": limit}

    def artifact(self, identifier, filename):
        if filename not in EXPORT_FILES:
            raise KeyError(filename)
        path = self.folder(identifier) / filename
        if path.is_symlink() or not path.is_file():
            raise KeyError(filename)
        return path
