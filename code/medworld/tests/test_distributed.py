from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from medworld.config import load_config
from medworld.batching import BatchPrefetch, training_finished, rank_slice
from medworld.ema import EMATarget
from medworld.model import MedWorld
from medworld.launch_distributed import worker_pids


class IndexData:
    def training_batch(self, task, offset, batch_size, seed):
        return {"task": task, "indices": list(range(offset, offset + batch_size))}


class BranchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 4)
        self.heads = nn.ModuleList([nn.Linear(4, 1), nn.Linear(4, 1)])

        self.world = nn.Linear(4, 4)
        self.target = EMATarget(self.encoder)
        self.cfg = load_config(overrides={"amp": False})

    forward = MedWorld.forward

    @property
    def device(self):
        return self.encoder.weight.device

    def current_loss(self, task, batch):
        state = self.encoder(batch["x"]).tanh()
        loss = (self.heads[task](state) - .25).square().mean()
        return loss, {"current": loss.detach()}

    def temporal_loss(self, batch):
        predicted = self.world(self.encoder(batch["x"]))
        loss = (predicted - self.target(batch["x"] + .3)).square().mean()
        return loss, {"latent": loss.detach()}


def ddp_worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=2)
    torch.manual_seed(8)
    model = BranchModel()
    ddp = DistributedDataParallel(model, find_unused_parameters=True, broadcast_buffers=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=.03, momentum=.9)
    target = model.target
    for step in range(4):
        optimizer.zero_grad(set_to_none=True)
        for micro in range(2):
            start = step * 8 + micro * 4 + rank * 2
            x = torch.arange(start * 3, (start + 2) * 3).float().reshape(2, 3) / 100
            if micro == 0:
                with ddp.no_sync():
                    (ddp(step % 2, {"x": x}, {"x": x + .1})[0] / 2).backward()
            else:
                (ddp(step % 2, {"x": x}, {"x": x + .1})[0] / 2).backward()
        optimizer.step()
        target.update(model.encoder, .9)
    torch.save({"model": model.state_dict(), "target": target.compact_state()}, Path(output) / f"rank{rank}.pt")
    dist.destroy_process_group()


class DistributedTests(unittest.TestCase):
    def test_worker_ownership_across_separate_process_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                                     start_new_session=True)
            try:
                (Path(tmp) / "rank0.pid").write_text(str(child.pid))
                self.assertNotEqual(os.getpgid(child.pid), os.getpgid(0))
                self.assertEqual(worker_pids(tmp, os.getpid()), [child.pid])
                self.assertEqual(worker_pids(tmp, child.pid), [])
            finally:
                child.terminate()
                child.wait(timeout=5)

    def test_shards_are_disjoint_and_advance_global_cursor(self):
        a, end_a = rank_slice(16, 4, 0, 2)
        b, end_b = rank_slice(16, 4, 1, 2)
        self.assertEqual((a, b, end_a, end_b), (16, 20, 24, 24))
        self.assertFalse(set(range(a, a + 4)) & set(range(b, b + 4)))

    def test_prefetch_does_not_commit_unconsumed_samples(self):
        cfg = load_config(overrides={"batch_size": 3, "accumulation": 2})
        progress = {"step": 0, "offsets": {t: 0 for t in
                    ("classification", "segmentation", "vqa", "temporal")}}
        streams = [BatchPrefetch(IndexData(), cfg, progress, rank, 2) for rank in range(2)]
        try:
            first, second = [s.next() for s in streams]
            self.assertEqual(first[0], "classification")
            self.assertEqual(first[1][0][0]["indices"], [0, 1, 2])
            self.assertEqual(second[1][0][0]["indices"], [3, 4, 5])
            self.assertEqual(first[1][1][0]["indices"], [6, 7, 8])
            self.assertEqual(first[2]["offsets"]["classification"], 12)
            self.assertEqual(first[2], second[2])
            self.assertTrue(all(value == 0 for value in progress["offsets"].values()))
        finally:
            for stream in streams:
                stream.close()
        resumed = {**progress, **first[2], "step": 1}
        stream = BatchPrefetch(IndexData(), cfg, resumed, 0, 2)
        try:
            task, batches, marker = stream.next()
            self.assertEqual(task, "segmentation")
            self.assertEqual(batches[0][0]["indices"], [0, 1, 2])
            self.assertEqual(marker["offsets"]["classification"], 12)
        finally:
            stream.close()

    def test_joint_sizes_preserve_independent_rank_offsets(self):
        cfg = load_config(overrides={"batch_size": 8, "accumulation": 1,
                                     "task_batch_sizes": {"classification": 2}})
        progress = {"step": 0, "offsets": {t: 0 for t in
                    ("classification", "segmentation", "vqa", "temporal")}}
        streams = [BatchPrefetch(IndexData(), cfg, progress, rank, 4) for rank in range(4)]
        try:
            samples = [s.next() for s in streams]
            for rank, (task, batches, marker) in enumerate(samples):
                self.assertEqual(task, "classification")
                self.assertEqual(batches[0][0]["indices"], list(range(rank * 2, rank * 2 + 2)))
                self.assertEqual(batches[0][1]["indices"], list(range(rank * 8, rank * 8 + 8)))
                self.assertEqual(marker["offsets"]["temporal"], 32)
                self.assertEqual(marker["offsets"]["classification"], 8)
            self.assertEqual(progress["offsets"]["classification"], 0)
        finally:
            for stream in streams:
                stream.close()

    def test_preprocessing_runs_in_prefetch_worker_without_advancing_cursor(self):
        import threading
        cfg = load_config(overrides={"batch_size": 2, "accumulation": 1})
        progress = {"step": 0, "offsets": {t: 0 for t in
                    ("classification", "segmentation", "vqa", "temporal")}}
        def prepare(task, batch):
            return {**batch, "prepared_task": task, "worker": threading.current_thread().name}
        stream = BatchPrefetch(IndexData(), cfg, progress, 1, 2, prepare=prepare)
        try:
            task, batches, marker = stream.next()
            current, temporal = batches[0]
            self.assertEqual(current["indices"], [2, 3])
            self.assertEqual(current["prepared_task"], task)
            self.assertEqual(temporal["prepared_task"], "temporal")
            self.assertTrue(current["worker"].startswith("data-rank1"))
            self.assertEqual(marker["offsets"]["classification"], 4)
            self.assertTrue(all(v == 0 for v in progress["offsets"].values()))
        finally:
            stream.close()

    def test_one_time_or_step_budget(self):
        cfg = load_config(overrides={"total_hours": 8})
        progress = {"step": 3, "deadline_unix": 240}
        self.assertFalse(training_finished(progress, cfg, 239))
        self.assertTrue(training_finished(progress, cfg, 240))
        cfg = load_config(overrides={"steps": 4})
        self.assertFalse(training_finished(progress, cfg, 999))
        progress["step"] = 4
        self.assertTrue(training_finished(progress, cfg, 0))
        for values in ({"stage1_hours": 4}, {"replay_every": 4}, {"task_batch_sizes": {"temporal": 0}}):
            with self.assertRaises(ValueError):
                load_config(overrides=values)

    def test_two_rank_dynamic_tasks_match_single_global_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(ddp_worker, args=(str(Path(tmp) / "rendezvous"), tmp), nprocs=2, join=True)
            torch.manual_seed(8)
            model = BranchModel()
            optimizer = torch.optim.SGD(model.parameters(), lr=.03, momentum=.9)
            target = model.target
            for step in range(4):
                optimizer.zero_grad(set_to_none=True)
                x = torch.arange(step * 24, (step + 1) * 24).float().reshape(8, 3) / 100
                model(step % 2, {"x": x}, {"x": x + .1})[0].backward()
                optimizer.step()
                target.update(model.encoder, .9)
            for rank in range(2):
                saved = torch.load(Path(tmp) / f"rank{rank}.pt", weights_only=True)
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, saved["model"][name], atol=1e-6, rtol=1e-5)
                self.assertEqual(saved["target"]["updates"], 4)
                for name, value in target.compact_state()["parameters"].items():
                    torch.testing.assert_close(value, saved["target"]["parameters"][name], atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
