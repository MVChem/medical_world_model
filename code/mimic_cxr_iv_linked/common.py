"""Shared local-only IO and deterministic cleaning rules."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
DEFAULT_CXR = Path('/home/data1/data/MIMIC/MIMIC_CXR')
DEFAULT_IV = Path('/home/data1/data/MIMIC/mimic-iv-3.1')
VERSION = 'cxr-iv-linked-v1'
FINDINGS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion', 'Pneumothorax']
HEADING = re.compile(r'(?m)^[ \t]*([A-Z][A-Z0-9 /_()\-]{1,60}):[ \t]*')


def csv_rows(path):
    path = Path(path)
    with (gzip.open if path.suffix == '.gz' else open)(path, 'rt', encoding='utf-8-sig', newline='') as f:
        yield from csv.DictReader(f)


def read_jsonl(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    os.chmod(tmp, 0o600)
    tmp.replace(path)


def dump_jsonl(path, rows):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    os.chmod(tmp, 0o600)
    tmp.replace(path)


def timestamp(value):
    return datetime.fromisoformat(value) if value else None


def iso(value):
    return value.isoformat(sep=' ') if value else None


def fingerprint(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def clean_report(raw):
    """Keep original section order/text, without duplicating a combined heading.

    No language model rewriting, clinical assertion removal or token truncation.
    Comparison/uncertainty flags identify review needs, not exclusion criteria.
    """
    text = raw.replace('\r\n', '\n').replace('\r', '\n')
    matches = list(HEADING.finditer(text))
    sections, seen = [], set()
    repeated = 0
    for i, match in enumerate(matches):
        name = ' '.join(match.group(1).upper().split())
        if name not in {'FINDINGS', 'IMPRESSION', 'FINDINGS AND IMPRESSION', 'FINDINGS/IMPRESSION'}:
            continue
        stop = matches[i+1].start() if i+1 < len(matches) else len(text)
        value = ' '.join(text[match.end():stop].split())
        if not value:
            continue
        key = (name, value)
        if key in seen:
            repeated += 1
            continue
        seen.add(key)
        sections.append(dict(name=name, text=value))
    cleaned = '\n'.join(s['name'] + ': ' + s['text'] for s in sections)
    counts = Counter(s['name'] for s in sections)
    return dict(text=cleaned, sections=sections, valid=bool(sections),
        raw_sha256=hashlib.sha256(raw.encode()).hexdigest(),
        clean_sha256=hashlib.sha256(cleaned.encode()).hexdigest(),
        chars=len(cleaned), words=len(cleaned.split()), token_truncated=False,
        report_available_time=None,
        flags=dict(no_supported_section=not bool(sections),
            repeated_identical_sections_removed=repeated,
            repeated_heading_different_content=any(v > 1 for v in counts.values()),
            combined_heading=any('AND' in s['name'] or '/' in s['name'] for s in sections),
            identical_findings_impression=bool(len(sections) == 2 and sections[0]['text'] == sections[1]['text']),
            references_other_observations=bool(re.search(r'\b(prior|previous|comparison|compared|interval|unchanged|CT)\b', cleaned, re.I)),
            uncertain_language=bool(re.search(r'\b(possible|possibly|cannot exclude|may represent|suspected|questionable)\b', cleaned, re.I)),
            long_report_over_256_words=len(cleaned.split()) > 256))


def containing_ids(t, intervals):
    if t is None:
        return []
    return sorted({key for key, start, end in intervals if start is not None and end is not None and start <= t <= end})


def link_status(a, b):
    if len(a) > 1 or len(b) > 1:
        return 'ambiguous_endpoint'
    if a and b:
        return 'same_admission' if a == b else 'different_admissions'
    return 'neither_linked' if not a and not b else 'one_unlinked'


def same_known(a, b):
    return bool(a and b and a.strip().upper() == b.strip().upper())


def horizon(hours):
    return '6-24h' if hours <= 24 else '>24-72h' if hours <= 72 else '>72-168h' if hours <= 168 else '>168-720h'


def available_time(event_time, store_time):
    """Recorded event proxy requires BOTH occurrence and recording timestamps."""
    a, b = timestamp(event_time), timestamp(store_time)
    return max(a, b) if a is not None and b is not None else None


def group_summary(rows):
    rows = list(rows)
    counts = Counter(r['subject_id'] for r in rows)
    return dict(pairs=len(rows), patients=len(counts),
        admissions=len({r['hadm_id'] for r in rows if r.get('hadm_id')}),
        views=dict(Counter(r['view'] for r in rows)), horizons=dict(Counter(r['horizon_bin'] for r in rows)),
        maximum_pairs_per_patient=max(counts.values(), default=0))
