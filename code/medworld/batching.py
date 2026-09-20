"""Deterministic joint batches shared by single-device and distributed trainers."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from .datasets import TASKS


def rank_slice(offset, batch_size, rank, world_size):
    if not 0 <= rank < world_size or offset < 0 or batch_size < 1:
        raise ValueError("Invalid distributed stream coordinates")
    return offset + rank * batch_size, offset + world_size * batch_size


def training_finished(progress, cfg, now):
    if cfg.get("total_hours", 0) > 0:
        return now >= progress["deadline_unix"]
    return progress["step"] >= cfg["steps"]


class BatchPrefetch:
    """Plan ahead without advancing the committed checkpoint cursor."""
    def __init__(self, data, cfg, progress, rank, world_size):
        self.data, self.cfg, self.rank, self.world_size = data, cfg, rank, world_size
        self.planned = dict(progress["offsets"])
        self.step = progress["step"]
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"data-rank{rank}")
        self.queue = deque()
        for _ in range(cfg.get("prefetch_batches", 2)):
            self._submit()

    def _request(self, task):
        size = self.cfg.get("task_batch_sizes", {}).get(task, self.cfg["batch_size"])
        offset, following = rank_slice(self.planned[task], size, self.rank, self.world_size)
        self.planned[task] = following
        return task, offset, size

    def _submit(self):
        task = TASKS[self.step % len(TASKS)]
        requests = [(self._request(task), self._request("temporal"))
                    for _ in range(self.cfg["accumulation"])]
        marker = {"offsets": dict(self.planned)}
        def load():
            def batch(request):
                return self.data.training_batch(*request, self.cfg["seed"]) if request is not None else None
            return task, [(batch(current), batch(temporal)) for current, temporal in requests], marker
        self.queue.append(self.pool.submit(load))
        self.step += 1

    def next(self):
        result = self.queue.popleft().result()
        self._submit()
        return result

    def close(self):
        for future in self.queue:
            future.cancel()
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.queue.clear()
