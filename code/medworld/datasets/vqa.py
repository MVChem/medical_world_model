"""Official CXR-VQA rows; source image and question are the only inputs."""
import json
from pathlib import Path
import ijson
from .protocol import _sha256

VOCABULARY_FILE = Path(__file__).with_name('vqa_vocabulary.json')
VOCABULARY = tuple(json.loads(VOCABULARY_FILE.read_text()))
INSTRUCTION = ('Answer using this chest radiograph. Return only a JSON array of concise answer labels; '
               'use ["yes"] or ["no"] for yes/no questions and [] for an empty answer.\nQuestion: ')


def load_vqa(cfg, records):
    hashes = {'vqa_vocabulary': _sha256(VOCABULARY_FILE)}
    for original, split in [('train', 'train'), ('valid', 'validate'), ('test', 'test')]:
        path = Path(cfg['vqa_data']) / (original + '.json')
        hashes[str(path)] = _sha256(path)
        seen = set()
        with path.open('rb') as handle:
            for row in ijson.items(handle, 'item'):
                identity = f'vqa:{original}:{row["idx"]}'
                if identity in seen or not set(row['answer']) <= set(VOCABULARY):
                    raise ValueError('Duplicate VQA ID or unknown reference label')
                seen.add(identity)
                records[split].append({'id': identity, 'subject_id': str(row['subject_id']),
                    'image': str(Path(cfg['image_root']) / row['image_path']), 'question': row['question'],
                    'answer': sorted(set(row['answer'])), 'semantic_type': row['semantic_type']})
    return hashes
