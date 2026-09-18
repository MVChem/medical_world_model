"""Extract UCSF into an atomic directory and verify every file against ZIP CRCs.

This command never deletes its source ZIP. Consumers must be checked before the
caller removes the original archive. Existing destination files are not replaced.
"""
import argparse
import hashlib
import json
import os
import shutil
import stat
import time
import zipfile
import zlib
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


def extract(archive, destination):
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    staging = destination.with_name('.' + destination.name + '.extracting')
    staging.mkdir(mode=0o700)  # Refuse to overwrite an earlier partial extraction.
    status_path = staging.parent / '.ucsf_alptdg_download/extraction_status.json'
    status_path.parent.mkdir(exist_ok=True)
    status = {"phase": "extracting", "archive": str(archive), "destination": str(destination)}

    def update(**fields):
        status.update(fields, updated_utc=datetime.now(timezone.utc).isoformat())
        temp = status_path.with_suffix('.tmp')
        temp.write_text(json.dumps(status, indent=2) + '\n')
        temp.replace(status_path)

    manifest = []
    start = time.monotonic()
    with zipfile.ZipFile(archive) as z:
        members = z.infolist()
        required = sum(x.file_size for x in members)
        if shutil.disk_usage(staging).free < required + 1024 ** 3:
            raise OSError("Insufficient free space for extraction")
        update(total_files=sum(not x.is_dir() for x in members), total_bytes=required,
               extracted_files=0, extracted_bytes=0)
        extracted_bytes = 0
        for member in members:
            rel = PurePosixPath(member.filename)
            if rel.is_absolute() or '..' in rel.parts or not rel.parts:
                raise ValueError(f"Unsafe archive path: {member.filename}")
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f"Symlink in archive: {member.filename}")
            target = staging.joinpath(*rel.parts)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member) as source, target.open('xb') as output:
                shutil.copyfileobj(source, output, length=2 ** 20)
            # Independent read of the written bytes, not just the input stream.
            crc, size, sha = 0, 0, hashlib.sha256()
            with target.open('rb') as f:
                for block in iter(lambda: f.read(2 ** 20), b''):
                    crc = zlib.crc32(block, crc)
                    size += len(block)
                    sha.update(block)
            if size != member.file_size or crc != member.CRC:
                raise IOError(f"Extracted file failed size/CRC verification: {member.filename}")
            manifest.append({"path": member.filename, "bytes": size,
                             "crc32": f"{crc:08x}", "sha256": sha.hexdigest()})
            extracted_bytes += size
            if len(manifest) % 100 == 0:
                update(extracted_files=len(manifest), extracted_bytes=extracted_bytes)
                print(f"Verified {len(manifest)}/{status['total_files']} files, {extracted_bytes/1e9:.2f} GB", flush=True)
    images = [m for m in manifest if m['path'].endswith('.nii.gz')]
    patients = {m['path'].split('/')[0] for m in images}
    if len(patients) != 298 or len(images) != 4768 or len(manifest) != 4769:
        raise ValueError("Unexpected UCSF patient/file counts; original ZIP retained")
    report = {"source_archive": archive.name, "verified_utc": datetime.now(timezone.utc).isoformat(),
              "verification": "Every extracted file independently reread: ZIP CRC32, size and SHA-256",
              "patients": len(patients), "image_files": len(images), "files": len(manifest),
              "bytes": extracted_bytes, "members": manifest}
    (staging / 'extraction_manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    # Flush file contents before publishing the directory as complete.
    os.sync()
    staging.rename(destination)
    update(phase="verified", extracted_files=len(manifest), extracted_bytes=extracted_bytes,
           patient_count=len(patients), image_files=len(images), elapsed_seconds=round(time.monotonic() - start, 1),
           source_archive_deleted=False, manifest=str(destination / 'extraction_manifest.json'))
    print(json.dumps({k: v for k, v in report.items() if k != 'members'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, default=Path('/home/data2/chk/data/UCSF_POSTOP_GLIOMA_DATASET_FINAL_v1.0.zip'))
    parser.add_argument('--destination', type=Path, default=Path('/home/data2/chk/data/UCSF-ALPTDG'))
    args = parser.parse_args()
    extract(args.archive, args.destination)
