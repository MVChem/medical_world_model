"""Inference reads frozen inputs only; references are opened only by the scorer."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

from .common import atomic, digest, read, rows, groups, key, prompt_for
from infer import request, image_url, probability_from_logprobs


def main(args):
    from transformers import AutoTokenizer
    os.umask(0o077)
    run = args.run.resolve()
    protocol = read(run / "protocol.json")
    # Verify only inputs here: inference does not read reference artifacts.
    for name, expected in protocol["file_sha256"].items():
        if name.startswith("images/") or "inputs_" in name or name == "models.json":
            if digest(run / name) != expected:
                raise ValueError(f"Frozen input changed: {name}")
    model = next(m for m in read(run / "models.json") if m["id"] == args.model)
    server = request(args.endpoint, route="/v1/models")["data"][0]
    if Path(server["root"]).resolve() != Path(model["path"]).resolve():
        raise ValueError("Wrong checkpoint at endpoint")
    tokenizer = AutoTokenizer.from_pretrained(model["path"], local_files_only=True)
    candidates = {word:tokenizer.encode(word, add_special_tokens=False) for word in ("Yes", "No")}
    if any(len(ids) != 1 for ids in candidates.values()):
        raise ValueError("Yes/No candidates must each be a single complete token")
    candidates = {word:ids[0] for word, ids in candidates.items()}
    out = run / args.model / ("smoke" if args.smoke else "test")
    out.mkdir(parents=True, exist_ok=True)
    plan = dict(protocol_sha256=digest(run / "protocol.json"), model=model, candidates=candidates,
                smoke=args.smoke, inference_sha256=digest(__file__), prompt_sha256=digest(Path(__file__).with_name("common.py")))
    if (out / "plan.json").exists() and read(out / "plan.json") != plan:
        raise ValueError("Resume plan changed")
    atomic(out / "plan.json", plan)
    journal = out / "responses.jsonl"
    if journal.exists():
        lines = journal.read_bytes().splitlines(keepends=True)
        if lines and not lines[-1].endswith(b"\n"):
            journal.write_bytes(b"".join(lines[:-1]))
    done = {r["key"]:r for r in rows(journal) if r.get("ok")} if journal.exists() else {}
    state = dict(status="running", pid=os.getpid(), model=args.model, started_unix=time.time(), tasks={})
    def flush():
        state["updated_unix"] = time.time()
        atomic(out / "status.json", state)
    def predict(task, row, finding):
        started = time.monotonic()
        result = dict(key=key(task, row, finding), task=task, id=row["id"], finding=finding, ok=False)
        prompt = prompt_for(task, row, finding)
        result["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
        probability = task.endswith("_prob")
        payload = dict(model=server["id"], messages=[dict(role="user", content=[
            dict(type="image_url", image_url=dict(url=image_url(row["image"]))), dict(type="text", text=prompt)])],
            temperature=0, seed=42, max_tokens=1 if probability else protocol["generation"]["max_new_tokens"])
        if model["family"] == "qwen":
            payload["chat_template_kwargs"] = {"enable_thinking":False}
        for attempt in range(3):
            try:
                if probability:
                    payload.update(allowed_token_ids=list(candidates.values()), logprobs=True, top_logprobs=20)
                response = request(args.endpoint, payload)
                choice = response["choices"][0]
                if probability:
                    token = choice["logprobs"]["content"][0]
                    lp = {x["token"]:x["logprob"] for x in token["top_logprobs"]}
                    lp[token["token"]] = token["logprob"]
                    for word, token_id in candidates.items():
                        if word not in lp:
                            extra = request(args.endpoint, dict(payload, allowed_token_ids=[token_id]))["choices"][0]["logprobs"]["content"][0]
                            if extra["token"] != word:
                                raise ValueError("Forced token mismatch")
                            lp[word] = extra["logprob"]
                    result.update(probability=probability_from_logprobs(lp["Yes"], lp["No"]),
                                  candidate_logprobs={word:lp[word] for word in candidates})
                    if args.smoke:
                        # Verify that allowed-token masks do not renormalize reported likelihoods.
                        plain = dict(payload)
                        plain.pop("allowed_token_ids")
                        raw = request(args.endpoint, plain)["choices"][0]["logprobs"]["content"][0]
                        raw_lp = {x["token"]:x["logprob"] for x in raw["top_logprobs"]}
                        raw_lp[raw["token"]] = raw["logprob"]
                        overlap = set(candidates) & set(raw_lp)
                        if not overlap or any(abs(raw_lp[w] - lp[w]) > 0.01 for w in overlap):
                            raise ValueError("Raw likelihood verification failed")
                        for word, token_id in candidates.items():
                            forced = request(args.endpoint, dict(payload, allowed_token_ids=[token_id]))["choices"][0]["logprobs"]["content"][0]
                            if forced["token"] != word or abs(forced["logprob"] - lp[word]) > 0.01:
                                raise ValueError("Forced token changed raw probability")
                        result["raw_likelihood_verified"] = True
                else:
                    result.update(text=choice["message"].get("content") or "", finish_reason=choice["finish_reason"])
                result.update(ok=True, attempts=attempt + 1, usage=response["usage"])
                break
            except Exception as error:
                result["error"] = f"{type(error).__name__}: {error}"
                if attempt < 2:
                    time.sleep(attempt + 1)
        result["seconds"] = time.monotonic() - started
        return result
    with journal.open("a", buffering=1) as handle:
        for task, entries in groups(run).items():
            if args.smoke:
                entries = entries[:2] if task == "table1_report" else entries[:1]
            pending = [(row, finding) for row, finding in entries if key(task, row, finding) not in done]
            state["current_task"] = task
            state["tasks"][task] = dict(expected=len(entries), completed=len(entries)-len(pending), errors=0)
            flush()
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                # Ordered chunks bound memory and permit durable restart.
                for start in range(0, len(pending), 32):
                    for result in pool.map(lambda pair, task=task:predict(task, *pair), pending[start:start+32]):
                        handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                        state["tasks"][task]["completed" if result["ok"] else "errors"] += 1
                    flush()
                    print(args.model, task, state["tasks"][task], flush=True)
    state["status"] = "complete" if all(v["expected"] == v["completed"] for v in state["tasks"].values()) else "failed"
    flush()
    if state["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--endpoint", required=True)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--smoke", action="store_true")
    main(p.parse_args())
