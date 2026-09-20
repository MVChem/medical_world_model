"""Read-only VQA dataset API, independent of patient clinical-table loading."""
from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from ..vqa import VQACatalog


def create_vqa_router(catalog=None):
    catalog = catalog or VQACatalog()
    router = APIRouter(prefix="/api/vqa", tags=["VQA"])

    def ready():
        state = catalog.status()
        if state["state"] != "ready":
            raise HTTPException(503, state.get("error") or "VQA 数据目录正在加载")

    @router.get("/summary")
    def summary():
        return catalog.status()

    @router.get("/questions")
    def questions(split: Literal["all", "train", "valid", "test"] = "all",
                  semantic: str = Query("all", max_length=100),
                  content: str = Query("all", max_length=100),
                  answers: Literal["all", "empty", "nonempty"] = "all",
                  q: str = Query("", max_length=200), image_id: str = Query("", max_length=100),
                  page: int = Query(1, ge=1), limit: int = Query(25, ge=1, le=100)):
        ready()
        return catalog.search(split=split, semantic=semantic, content=content, answers=answers,
                              q=q, image_id=image_id, page=page, limit=limit)

    @router.get("/questions/{split}/{position}")
    def question(split: Literal["train", "valid", "test"], position: int):
        ready()
        if position < 0 or position >= len(catalog.rows[split]):
            raise HTTPException(404, "Unknown VQA question")
        return catalog.detail(split, position)

    return router
