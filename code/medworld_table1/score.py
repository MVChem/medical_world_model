"""Clinical scoring subprocess with a separate Transformers 4 dependency path."""
import os
from pathlib import Path
import runpy
import sys

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / 'metric_vendor'))
os.environ['MEDWORLD_METRIC_WORKER'] = '1'
runpy.run_path(str(root / 'evaluate.py'), run_name='__main__')
