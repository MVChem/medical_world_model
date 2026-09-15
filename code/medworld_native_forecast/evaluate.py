"""Source-only held-out generation in the existing Table-1 scorer's schema.

Output predictions.jsonl contains {id, report, scores}; scores are the supervised
finding head's sigmoid probabilities. Clinical metrics are scored separately.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time

import torch

from run import FORMAT_VERSION, atomic_json, autocast, check_runtime, data_signature, digest, runtime_signature, seed_all


def atomic_rows(path, rows):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def evaluate(args):
    from data import Corpus
    from model import NativeForecast
    os.umask(0o077)
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError(f"Evaluation output exists; use --resume for the same checkpoint: {out}")
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["format_version"] != FORMAT_VERSION or checkpoint["stage"] != 2 or not checkpoint["completed_stage"]:
        raise ValueError("Evaluation requires a completed Stage-2 checkpoint")
    if checkpoint["smoke_only"] and not args.allow_smoke:
        raise ValueError("Smoke checkpoints cannot produce formal Table-1 evaluations")
    cfg = checkpoint["config"]
    check_runtime(checkpoint, runtime_signature(cfg), smoke=args.allow_smoke)
    seed_all(cfg["seed"])
    torch.set_num_threads(args.threads)
    model = NativeForecast(cfg, condition=checkpoint["condition"], device=args.device)
    model.load_compact(checkpoint["model"])
    model.eval()
    corpus = Corpus(cfg, model.tokenizer)
    if checkpoint["data_metadata"] != data_signature(corpus, cfg):
        raise ValueError("Evaluation data provenance differs from training")
    rows = corpus.pairs[args.split][:args.limit or None]
    contract = dict(checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_sha256=digest(args.checkpoint),
        condition=checkpoint["condition"], split=args.split, n=len(rows), limit=args.limit,
        max_new_tokens=cfg["generation_tokens"], smoke_only=checkpoint["smoke_only"],
        probability_source="supervised finding head sigmoid", decoding="greedy", teacher_forcing=False,
        target_inputs=False, input_protocol="current raw image + current report/prior EHR + horizon; additional predicted future slots when enabled",
        pair_ids_sha256=__import__("hashlib").sha256(json.dumps([row["id"] for row in rows]).encode()).hexdigest())
    contract_path = out / "generation_contract.json"
    if args.resume:
        if not contract_path.is_file() or json.loads(contract_path.read_text()) != contract:
            raise ValueError("Evaluation resume contract differs")
    atomic_json(contract_path, contract)
    atomic_json(out / "config.json", cfg)
    partial = out / "predictions.partial.jsonl"
    generated = [json.loads(line) for line in partial.read_text().splitlines() if line.strip()] if args.resume and partial.exists() else []
    if [row["id"] for row in generated] != [row["id"] for row in rows[:len(generated)]]:
        raise ValueError("Partial prediction IDs/order differ from evaluation cohort")
    started = time.monotonic()
    del checkpoint
    with torch.inference_mode():
        for row in rows[len(generated):]:
            # A pair without any target identifier must remain fully predictable.
            source_row = {key: value for key, value in row.items() if key != "target"}
            batch = corpus.batch([source_row], device=args.device, source_only=True)
            with autocast(args.device):
                reports, scores = model.predict(batch)
            scores = scores.detach().float().cpu()
            if len(reports) != 1 or scores.shape != (1, len(cfg["findings"])):
                raise ValueError("Prediction batch dimensions differ from cohort/finding protocol")
            if not torch.isfinite(scores).all() or bool(((scores < 0) | (scores > 1)).any()):
                raise FloatingPointError("Finding head did not return finite probabilities")
            generated.append(dict(id=row["id"], report=reports[0], scores=scores[0].tolist()))
            atomic_rows(partial, generated)
            if len(generated) % 10 == 0 or len(generated) == len(rows):
                progress = dict(generated=len(generated), expected=len(rows), seconds=time.monotonic() - started)
                atomic_json(out / "generation_progress.json", progress)
                print(json.dumps(progress), flush=True)
    atomic_rows(out / "predictions.jsonl", generated)
    counts = Counter(row["report"] for row in generated)
    atomic_json(out / "diversity.json", dict(n=len(generated), unique_reports=len(counts),
        most_common_count=max(counts.values(), default=0), empty_reports=sum(not row["report"].strip() for row in generated)))
    atomic_json(out / "generation.json", dict(**contract, count=len(generated), state="complete",
        table1_ready=False, pending="Clinical scoring (CheXbert, RadGraph, GREEN) and reference coverage audit"))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", choices=("validate", "test"), default="test")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-smoke", action="store_true", help="Diagnostic generation only; never a formal table result")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.limit < 0 or args.threads <= 0:
        parser.error("Limit must be nonnegative and threads positive")
    return evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
