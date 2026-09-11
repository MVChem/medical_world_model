import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
# Third-party packages remain at their existing runtime location.
PILOT = PROJECT / 'code' / 'medworld_table1'
sys.path.insert(0, str(PILOT / 'vendor'))
sys.path.insert(0, str(PROJECT / 'code'))
sys.path.insert(0, str(ROOT))  # A run's frozen shared package takes precedence.
from medworld_common.runtime import atomic_json, atomic_torch, digest, seed_all, load_rows, write_rows

FINDINGS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
            'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
            'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']
ORGANS = ['right lung', 'left lung', 'heart']
