"""Read registered experiment status; no GPU allocation or training side effects."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def collect(include_history=False):
    entries = json.loads((ROOT / "experiments/registry.json").read_text())[
        "experiments"
    ]
    rows = []
    for entry in entries:
        if not include_history and entry["role"] != "current":
            continue
        path = ROOT / entry["run"] / entry["status"]
        try:
            state = json.loads(path.read_text())
        except FileNotFoundError:
            state = {"phase": "status file missing"}
        except json.JSONDecodeError:
            state = {"phase": "status unreadable; check file"}
        rows.append(
            dict(
                id=entry["id"],
                method=entry["method"],
                phase=state.get("phase", state.get("state", "unknown")),
                step=state.get("step"),
                pid=state.get("pid"),
                updated=state.get("updated", state.get("updated_unix")),
                status=str(path.relative_to(ROOT)),
            )
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all", action="store_true", help="Include historical experiments"
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    rows = collect(args.all)
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        for row in rows:
            print(
                f"{row['id']:<25} {row['phase']:<20} step={row['step']}  pid={row['pid']}"
            )
            print(f"  {row['status']}")


if __name__ == "__main__":
    main()
