"""Local utilities. No MIMIC-derived content is sent to a remote service."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'vendor'))

sys.path.insert(0, str(ROOT.parent))
from medworld_common.runtime import atomic_json, atomic_torch, digest, seed_all, load_rows, write_rows

def read_config(path):
    cfg = json.loads(Path(path).read_text())
    for key in ('qwen', 'vjepa_checkpoint', 'data_root', 'cache'):
        p = Path(cfg[key])
        cfg[key] = str(p if p.is_absolute() else ROOT / p)
    return cfg
