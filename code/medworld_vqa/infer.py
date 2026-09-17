"""Image/question-only inference; never opens reference answers or reports."""

import argparse
import base64
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

from .common import atomic, digest, inference_prompt, read, request, rows


@lru_cache(maxsize=512)
def image_url(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def main(args):
    os.umask(0o077)
    run = args.run.resolve()
    protocol = read(run / "protocol.json")
    spec = next(m for m in read(run / "models.json") if m["id"] == args.model)
    server = request(args.endpoint, route="/v1/models")["data"][0]
    if Path(server["root"]).resolve() != Path(spec["path"]).resolve():
        raise ValueError("Endpoint checkpoint mismatch")
    inputs = rows(run / "inputs.jsonl")
    vocabulary = read(run / "vocabulary.json")
    out = run / args.model
    out.mkdir(exist_ok=True)
    plan = {
        "protocol_sha256": digest(run / "protocol.json"),
        "source_sha256": digest(__file__),
        "prompt_source_sha256": digest(Path(__file__).with_name("common.py")),
        "model": spec,
        "input_sha256": digest(run / "inputs.jsonl"),
        "vocabulary_sha256": digest(run / "vocabulary.json"),
    }
    if (out / "plan.json").exists() and read(out / "plan.json") != plan:
        raise ValueError("Resume protocol changed")
    atomic(out / "plan.json", plan)
    journal = out / "predictions.jsonl"
    done = {}
    if journal.exists():
        raw = journal.read_bytes()
        if raw and not raw.endswith(b"\n"):
            journal.write_bytes(raw[: raw.rfind(b"\n") + 1])
        for r in rows(journal):
            if r["ok"]:
                done[r["id"]] = r
    if not set(done) <= {r["id"] for r in inputs}:
        raise ValueError("Unexpected resumed sample ID")
    start = time.monotonic()
    status = {
        "status": "running",
        "model": args.model,
        "n": len(inputs),
        "completed": len(done),
        "errors": 0,
        "pid": os.getpid(),
        "started_unix": time.time(),
    }
    atomic(out / "status.json", status)

    def predict(row):
        start = time.monotonic()
        payload = {
            "model": server["id"],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url(row["image"])},
                        },
                        {
                            "type": "text",
                            "text": inference_prompt(
                                row["question"],
                                vocabulary,
                                protocol.get("prompt_style"),
                            ),
                        },
                    ],
                }
            ],
            "temperature": 0,
            "seed": protocol["seed"],
            "max_tokens": protocol["generation"]["max_tokens"],
        }
        if spec["family"] == "qwen":
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        if protocol["generation"].get("structured_outputs"):
            payload["structured_outputs"] = protocol["generation"]["structured_outputs"]
        result = {"id": row["id"], "ok": False}
        for attempt in range(3):
            try:
                response = request(args.endpoint, payload)
                choice = response["choices"][0]
                result.update(
                    ok=True,
                    text=choice["message"].get("content") or "",
                    finish_reason=choice["finish_reason"],
                    usage=response["usage"],
                )
                break
            except Exception as e:  # noqa: BLE001 -- preserve per-request failures for scoring
                result["error"] = f"{type(e).__name__}: {e}"
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
        result.update(attempts=attempt + 1, seconds=time.monotonic() - start)
        return result

    with (
        journal.open("a", buffering=1) as log,
        ThreadPoolExecutor(max_workers=args.concurrency) as pool,
    ):
        futures = [pool.submit(predict, r) for r in inputs if r["id"] not in done]
        for future in as_completed(futures):
            result = future.result()
            log.write(json.dumps(result, ensure_ascii=False) + "\n")
            if result["ok"]:
                done[result["id"]] = result
            else:
                status["errors"] += 1
            status.update(
                completed=len(done),
                updated_unix=time.time(),
                seconds=time.monotonic() - start,
            )
            if (len(done) + status["errors"]) % 32 == 0:
                atomic(out / "status.json", status)
                print(json.dumps(status), flush=True)
        os.fsync(log.fileno())
    status.update(
        status="complete" if len(done) == len(inputs) else "failed",
        finished_unix=time.time(),
    )
    atomic(out / "status.json", status)
    if status["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--endpoint", required=True)
    p.add_argument("--concurrency", type=int, default=8)
    main(p.parse_args())
