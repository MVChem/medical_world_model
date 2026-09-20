"""Lazy, in-memory VQA catalog. Read source JSON; never persist data copies."""
from collections import Counter
from pathlib import Path
from threading import Lock, Thread

import ijson

from .data import DATA_ROOT

SPLITS = ("train", "valid", "test")
FIELDS = ("idx", "subject_id", "study_id", "image_id", "question", "semantic_type", "content_type", "answer")


class VQACatalog:
    def __init__(self, root=None):
        self.root = Path(root or DATA_ROOT / "MIMIC_CXR_VQA/MIMIC-Ext-MIMIC-CXR-VQA/dataset")
        self.lock = Lock()
        self.state = "idle"
        self.error = ""
        self.rows = {}
        self.summary = {}

    def status(self):
        with self.lock:
            if self.state == "idle":
                self.state = "loading"
                Thread(target=self._load, daemon=True, name="vqa-catalog").start()
            return {"state": self.state, "error": self.error, **self.summary}

    def _load(self):
        try:
            rows, stats, patients = {}, {}, {}
            for split in SPLITS:
                current = []
                images, subjects = set(), set()
                semantic, content = Counter(), Counter()
                empty = 0
                with (self.root / f"{split}.json").open("rb") as source:
                    for item in ijson.items(source, "item"):
                        row = tuple(tuple(item[k]) if k == "answer" else item[k] for k in FIELDS)
                        current.append(row)
                        subjects.add(row[1])
                        images.add(row[3])
                        semantic[row[5]] += 1
                        content[row[6]] += 1
                        empty += not row[7]
                rows[split] = current
                patients[split] = subjects
                stats[split] = {"questions": len(current), "images": len(images), "patients": len(subjects), "empty_answers": empty, "semantic_types": dict(semantic), "content_types": dict(content)}
            summary = {"splits": stats, "questions": sum(len(r) for r in rows.values()), "patients": len(set.union(*patients.values())), "images": len({r[3] for group in rows.values() for r in group}), "overlap": {f"{a}/{b}": len(patients[a] & patients[b]) for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))}}
            with self.lock:
                self.rows, self.summary, self.state = rows, summary, "ready"
        except Exception as exc:
            with self.lock:
                self.error = f"VQA source unavailable: {exc}"
                self.state = "error"

    @staticmethod
    def serialize(split, position, row):
        return {"id": f"{split}:{position}", "split": split, **dict(zip(FIELDS, row))}

    def search(self, *, split="all", semantic="all", content="all", answers="all", q="", image_id="", page=1, limit=25):
        query = q.strip().casefold()
        total, result = 0, []
        start = (page - 1) * limit
        for name in SPLITS if split == "all" else (split,):
            for pos, row in enumerate(self.rows[name]):
                if semantic != "all" and row[5] != semantic:
                    continue
                if content != "all" and row[6] != content:
                    continue
                if answers == "empty" and row[7] or answers == "nonempty" and not row[7]:
                    continue
                if image_id and row[3] != image_id:
                    continue
                if query and query not in " ".join((str(row[0]), row[1], row[2], row[3], row[4], *row[7])).casefold():
                    continue
                if start <= total < start + limit:
                    result.append(self.serialize(name, pos, row))
                total += 1
        return {"rows": result, "total": total, "page": page, "limit": limit}

    def detail(self, split, position):
        return self.serialize(split, position, self.rows[split][position])
