"""Validate the local datasets and create non-destructive machine-local links."""

from __future__ import annotations

import argparse
from pathlib import Path


def setup_links(source: Path, destination: Path) -> dict[str, Path]:
    source = source.expanduser().resolve()
    paths = {
        "MIMIC": source,
        "MIMIC_CXR": source / "MIMIC_CXR",
        "mimic-iv-3.1": source / "mimic-iv-3.1",
    }
    metadata = paths["MIMIC_CXR"] / "mimic-cxr-2.0.0-metadata.csv"
    if not metadata.is_file() and not metadata.with_suffix(".csv.gz").is_file():
        raise FileNotFoundError(f"CXR metadata not found under {paths['MIMIC_CXR']}")
    for table in ("hosp/admissions.csv.gz", "icu/icustays.csv.gz"):
        path = paths["mimic-iv-3.1"] / table
        if not path.is_file() and not path.with_suffix("").is_file():
            raise FileNotFoundError(f"MIMIC-IV table missing: {table}")
    destination = destination.expanduser().absolute()
    # Check the entire plan before creating any links.
    for name, target in paths.items():
        link = destination / name
        if (link.exists() or link.is_symlink()) and link.resolve() != target:
            raise FileExistsError(f"Refusing to replace existing path: {link}")
    destination.mkdir(parents=True, exist_ok=True)
    for name, target in paths.items():
        link = destination / name
        if not link.exists() and not link.is_symlink():
            link.symlink_to(target, target_is_directory=True)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("/home/data2/chk/data/MIMIC")
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data"
    )
    args = parser.parse_args()
    for name, path in setup_links(args.source, args.data_dir).items():
        print(f"{args.data_dir / name} -> {path}")


if __name__ == "__main__":
    main()
