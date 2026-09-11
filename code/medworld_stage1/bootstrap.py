import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
PILOT = ROOT.parent / 'medworld_table1'
sys.path.insert(0, str(PILOT))
sys.path.insert(0, str(PILOT / 'vendor'))
from common import atomic_json, atomic_torch, digest, seed_all, load_rows, write_rows

FINDINGS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
            'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
            'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']
ORGANS = ['right lung', 'left lung', 'heart']
