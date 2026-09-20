"""VQA source semantics, exact joins and API pagination boundaries."""
import json
import time

from fastapi import FastAPI
from fastapi.testclient import TestClient

from mimic_atlas.backend.vqa import create_vqa_router
from mimic_atlas.vqa import VQACatalog


def row(idx, subject, image, answer, semantic="query", content="abnormality"):
    return dict(idx=idx, subject_id=subject, study_id="study-1", image_id=image,
                question="What is visible?", semantic_type=semantic,
                content_type=content, answer=answer)


def catalog(tmp_path):
    sources = {
        "train": [row(0, "100", "image-a", ["effusion"]), row(1, "200", "image-b", []), row(2, "100", "image-a", ["yes"], "verify", "presence")],
        "valid": [row(0, "100", "image-c", ["no"], "verify", "presence")],
        "test": [row(0, "300", "image-d", ["left", "right"])],
    }
    for split, items in sources.items():
        (tmp_path / f"{split}.json").write_text(json.dumps(items))
    result = VQACatalog(tmp_path)
    result._load()
    assert result.state == "ready"
    return result


def test_counts_preserve_official_overlap_and_empty_sets(tmp_path):
    c = catalog(tmp_path)
    s = c.status()
    assert s["questions"] == 5 and s["images"] == 4 and s["patients"] == 3
    assert s["overlap"] == {"train/valid": 1, "train/test": 0, "valid/test": 0}
    assert s["splits"]["train"]["empty_answers"] == 1
    assert c.search(answers="empty")["rows"][0]["answer"] == ()
    assert c.search(answers="nonempty")["total"] == 4
    assert c.search(split="train", semantic="verify", content="presence")["total"] == 1
    assert c.search(q="EFFUSION")["total"] == 1
    assert c.search(image_id="image-a")["total"] == 2
    assert c.search(image_id="image")["total"] == 0
    assert c.search(q="absent")["total"] == 0


def test_api_pagination_stable_ids_and_validation(tmp_path):
    app = FastAPI()
    app.include_router(create_vqa_router(catalog(tmp_path)))
    with TestClient(app) as client:
        first = client.get("/api/vqa/questions?limit=2").json()
        second = client.get("/api/vqa/questions?limit=2&page=2").json()
        assert first["total"] == second["total"] == 5
        assert {r["id"] for r in first["rows"]}.isdisjoint(r["id"] for r in second["rows"])
        assert client.get("/api/vqa/questions/train/1").json()["answer"] == []
        assert client.get("/api/vqa/questions/train/-1").status_code == 404
        assert client.get("/api/vqa/questions/train/100").status_code == 404
        for query in ("split=validate", "page=0", "limit=101", "answers=missing"):
            assert client.get("/api/vqa/questions?" + query).status_code == 422
        assert client.get("/api/vqa/questions?page=100").json()["rows"] == []


def test_missing_source_is_explicit_and_does_not_publish_partial_data(tmp_path):
    c = VQACatalog(tmp_path)
    c.status()
    for _ in range(100):
        if c.status()["state"] == "error":
            break
        time.sleep(.01)
    assert c.status()["state"] == "error"
    assert c.rows == {}
    app = FastAPI()
    app.include_router(create_vqa_router(c))
    with TestClient(app) as client:
        assert client.get("/api/vqa/questions").status_code == 503
