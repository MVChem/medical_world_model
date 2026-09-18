"""Finish the MU download workflow: audit, test, refresh the live and offline viewers."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .data import DATA_ROOT

WORK = DATA_ROOT / ".mu_glioma_post_download"
DATASET = DATA_ROOT / "MU-Glioma-Post"
PROJECT = Path(__file__).resolve().parents[2]


def save(state):
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    temp = WORK / "finalization_status.json.tmp"
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    temp.replace(WORK / "finalization_status.json")


def main():
    state = {"phase": "waiting_for_verified_download", "steps": []}
    save(state)
    try:
        while True:
            status = json.loads((WORK / "status.json").read_text())
            if status["phase"] == "complete":
                break
            service = subprocess.run(["systemctl", "--user", "is-active", "mu-glioma-post-download.service"],
                                     capture_output=True, text=True, timeout=10)
            if status["phase"] == "incomplete" or service.stdout.strip() not in ("active", "activating"):
                raise RuntimeError("Download stopped before verified completion; see status.json")
            time.sleep(20)
        manifest = json.loads((DATASET / "image_manifest.json").read_text())
        if manifest["verified_files"] != 2978 or manifest["verified_bytes"] != 11890059719:
            raise ValueError("Unexpected final mirror totals")
        files = list(DATASET.glob("PatientID_*/Timepoint_*/*.nii.gz"))
        if len(files) != 2978 or sum(path.stat().st_size for path in files) != 11890059719:
            raise ValueError("Local files differ from verified manifest")
        expected = {row["path"] for row in manifest["files"]}
        if expected != {path.relative_to(DATASET).as_posix() for path in files}:
            raise ValueError("Local path set differs from verified manifest")
        audit = json.loads((DATASET / "matching_audit.json").read_text())
        audit.update(scope="Complete local download: all 2,978 files verified against pinned mirror SHA-256; path and size inventory also matches TCIA",
                     local_files_verified=True, mirror_revision=manifest["revision"])
        (DATASET / "matching_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
        state["steps"].append("Local paths, counts and total bytes match the verified image manifest")
        state["phase"] = "validating_and_refreshing_viewers"
        save(state)
        env = dict(os.environ, PYTHONPATH=str(PROJECT / "code"))
        commands = [
            ("Data, geometry, real MU timepoint and missing-mask tests",
             [sys.executable, "-m", "unittest", "glioma_explorer.test_data", "-v"]),
            ("Export self-contained HTML with complete cohort counts and two real MRI examples",
             [sys.executable, "-m", "glioma_explorer.export_html"]),
            ("Reload live viewer with all downloaded files",
             ["systemctl", "--user", "restart", "glioma-atlas.service"]),
            ("Browser checks: actual MRI, timepoint selection, mobile and offline HTML",
             [sys.executable, "-m", "glioma_explorer.browser_check"]),
        ]
        for label, command in commands:
            print(label, flush=True)
            subprocess.run(command, cwd=PROJECT, env=env, check=True, timeout=900)
            state["steps"].append(label)
            save(state)
        state["phase"] = "complete"
        state["verified_files"] = manifest["verified_files"]
        state["verified_bytes"] = manifest["verified_bytes"]
        save(state)
        status_path = DATASET / "DOWNLOAD_STATUS.md"
        with status_path.open("a") as stream:
            stream.write(f"\n最终完成：{state['updated_at']}（UTC）。全部 2,978 个 NIfTI 已校验；数据测试、浏览器检查通过，在线页面与离线 HTML 已刷新。\n")
        print(json.dumps(state, ensure_ascii=False), flush=True)
    except Exception as error:
        state.update(phase="failed", error=str(error))
        save(state)
        raise


if __name__ == "__main__":
    main()
