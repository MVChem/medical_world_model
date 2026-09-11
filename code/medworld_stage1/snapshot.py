"""Freeze executable sources, including shared modules, for each new run."""

from pathlib import Path
import shutil
import medworld_common
from bootstrap import ROOT, PROJECT, atomic_json, digest


def source_hashes(root):
    root = Path(root)
    result = {p.name: digest(p) for p in root.glob("*.py")}
    shared = root / "medworld_common"
    if not shared.exists():
        shared = Path(medworld_common.__file__).parent
    result.update({f"medworld_common/{p.name}": digest(p) for p in shared.glob("*.py")})
    return result


def create_snapshot(run):
    run = Path(run)
    source = run / "source"
    source.mkdir(parents=True, exist_ok=False)
    for path in ROOT.glob("*.py"):
        shutil.copy2(path, source / path.name)
    shutil.copytree(
        Path(medworld_common.__file__).parent,
        source / "medworld_common",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    path = source / "bootstrap.py"
    path.write_text(
        path.read_text().replace(
            "PROJECT = ROOT.parent.parent", f"PROJECT = Path({str(PROJECT)!r})"
        )
    )
    atomic_json(run / "source_manifest.json", source_hashes(source))
    return source
