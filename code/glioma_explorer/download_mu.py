"""Download the public MU mirror at a pinned revision; verify every LFS SHA-256.

MRI files go directly into DATA_ROOT/MU-Glioma-Post (no outer archive).
The third-party mirror is explicitly recorded, independently of TCIA provenance.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import threading
import time

import nibabel as nib
import requests

from .data import DATA_ROOT

REPOSITORY = "sbandred/mu-glioma-post-raw"
REVISION = "f6acd4d7d19d35304dc4317d9af4bd25094eb6b9"
WORK = DATA_ROOT / ".mu_glioma_post_download"
DESTINATION = DATA_ROOT / "MU-Glioma-Post"
LOCAL = threading.local()


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def session():
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
    return LOCAL.session


def verify(path, item):
    if not path.is_file() or path.stat().st_size != item["size"]:
        return False
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != item["lfs"]["oid"]:
            return False
    return True


def fetch(item):
    relative = PurePosixPath(item["path"])
    if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "MU-Glioma-Post":
        raise ValueError("Unexpected mirror path")
    destination = DATA_ROOT.joinpath(*relative.parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{relative.as_posix()}"
    last_error = ""
    for attempt in range(8):
        try:
            if not verify(destination, item):
                offset = partial.stat().st_size if partial.exists() else 0
                if offset >= item["size"]:
                    partial.unlink()
                    offset = 0
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                with session().get(url, headers=headers, stream=True, timeout=(45, 90)) as response:
                    response.raise_for_status()
                    resume = offset > 0 and response.status_code == 206
                    if resume and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise ValueError("Invalid range response")
                    with partial.open("ab" if resume else "wb") as stream:
                        for chunk in response.iter_content(1024 * 1024):
                            stream.write(chunk)
                if not verify(partial, item):
                    partial.unlink(missing_ok=True)
                    raise ValueError("Downloaded size or SHA-256 mismatch")
                partial.replace(destination)
            # Reading to EOF checks gzip CRC, beyond checking the compressed SHA.
            with gzip.open(destination, "rb") as stream:
                while stream.read(1024 * 1024):
                    pass
            image = nib.load(destination)
            if len(image.shape) != 3 or any(size < 1 for size in image.shape):
                raise ValueError("Invalid 3D NIfTI shape")
            return {"path": relative.relative_to("MU-Glioma-Post").as_posix(),
                    "bytes": item["size"], "sha256": item["lfs"]["oid"],
                    "shape": list(image.shape), "spacing": list(map(float, image.header.get_zooms())),
                    "spatial_unit": image.header.get_xyzt_units()[0],
                    "gzip_crc_verified": True}
        except Exception as error:
            # HTTP exceptions can include temporary signed URLs; do not log them.
            last_error = type(error).__name__
            if attempt < 7:
                time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"{relative.as_posix()}: {last_error} after eight attempts")


def main(workers):
    WORK.mkdir(parents=True, exist_ok=True)
    DESTINATION.mkdir(parents=True, exist_ok=True)
    if not (WORK / "hf_repository.json").exists() or not (WORK / "hf_tree.json").exists():
        api = f"https://huggingface.co/api/datasets/{REPOSITORY}"
        response = session().get(f"{api}/revision/{REVISION}", timeout=(45, 90))
        response.raise_for_status()
        repository = response.json()
        tree, url = [], f"{api}/tree/{REVISION}?recursive=true&expand=false&limit=1000"
        while url:
            response = session().get(url, timeout=(45, 90))
            response.raise_for_status()
            tree.extend(response.json())
            url = response.links.get("next", {}).get("url")
        write_json(WORK / "hf_repository.json", repository)
        write_json(WORK / "hf_tree.json", tree)
    repository = json.loads((WORK / "hf_repository.json").read_text())
    if repository["sha"] != REVISION:
        raise ValueError("Cached repository revision differs from the pinned revision")
    tree = json.loads((WORK / "hf_tree.json").read_text())
    files = [item for item in tree if item["type"] == "file" and item["path"].endswith(".nii.gz")]
    for item in files:
        if item["size"] != item["lfs"]["size"]:
            raise ValueError("Invalid LFS manifest")
    started = time.time()
    status = {"phase": "downloading", "started_at": timestamp(), "updated_at": timestamp(),
              "source": f"https://huggingface.co/datasets/{REPOSITORY}", "source_type": "third_party_mirror",
              "revision": REVISION, "destination": str(DESTINATION),
              "total_files": len(files), "total_bytes": sum(item["size"] for item in files),
              "verified_files": 0, "verified_bytes": 0, "errors": []}
    write_json(WORK / "status.json", status)
    verified = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch, item): item for item in files}
        for future in as_completed(futures):
            try:
                record = future.result()
                verified.append(record)
                status["verified_files"] += 1
                status["verified_bytes"] += record["bytes"]
            except Exception as error:
                status["errors"].append(str(error))
            status.update(updated_at=timestamp(), elapsed_seconds=round(time.time() - started, 1))
            write_json(WORK / "status.json", status)
            if status["verified_files"] % 50 == 0:
                print(f"Verified {status['verified_files']}/{len(files)} files, "
                      f"{status['verified_bytes'] / 1e9:.3f}/{status['total_bytes'] / 1e9:.3f} GB", flush=True)
    status["phase"] = "complete" if not status["errors"] and len(verified) == len(files) else "incomplete"
    status["completed_at"] = timestamp()
    manifest = {**status, "official_collection": "https://www.cancerimagingarchive.net/collection/mu-glioma-post/",
                "verification": "Every compressed file SHA-256 equals pinned mirror LFS OID; gzip CRC checked; 3D NIfTI header readable. This does not establish full official TCIA byte equivalence.",
                "files": sorted(verified, key=lambda record: record["path"])}
    write_json(DESTINATION / "image_manifest.json", manifest)
    write_json(WORK / "status.json", status)
    print(json.dumps({key: value for key, value in status.items() if key != "errors"}), flush=True)
    if status["phase"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    main(parser.parse_args().workers)
