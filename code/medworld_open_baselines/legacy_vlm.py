"""Native CheXagent-8b / LLaVA-Med inference with the frozen table prompts.

Run with the isolated Transformers 4.36.2 dependencies installed in legacy_vendor.
Only source/current images and prepared input JSONL are read; no references are
passed to either model. Probabilities are from unmodified next-token logits.
"""
from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PROJECT = Path(os.environ.get("MEDWORLD_PROJECT", HERE.parent.parent)).resolve()
BASE = PROJECT / "code/medworld_open_baselines"


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--model", choices=["chexagent8b", "llava_med7b"], required=True)
    p.add_argument("--assets", type=Path)
    p.add_argument("--model-path", type=Path, help="Explicit local snapshot, useful for metadata inspection")
    p.add_argument("--clip-path", type=Path)
    p.add_argument("--vendor", type=Path, default=BASE / "legacy_vendor")
    p.add_argument("--llava-repo", type=Path, default=BASE / "third_party/LLaVA-Med")
    p.add_argument("--source", type=Path, help="Frozen medworld_baselines source directory")
    p.add_argument("--split", choices=["test", "validate"], default="test")
    p.add_argument("--tasks", default="all")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-context", type=int, default=8192)
    p.add_argument("--inspect", action="store_true", help="CPU metadata/processor checks; no model weights loaded")
    return p.parse_args()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    tmp.replace(path)


def resolve_asset(args, name):
    explicit = args.model_path if name == args.model else args.clip_path
    if explicit:
        return explicit.resolve(), dict(status="explicit_local_snapshot", path=str(explicit.resolve()))
    if args.assets:
        status = json.loads((args.assets / name / "download_status.json").read_text())
        if status["status"] != "complete":
            raise RuntimeError(f"{name} weights not ready: {status['status']}")
        return Path(status["path"]), status
    inventory = json.loads((args.run / "models.json").read_text())
    model = next(m for m in inventory if m["id"] == name)
    return Path(model["path"]), model


class NativeModel:
    def __init__(self, args):
        import torch
        import transformers
        from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer, CLIPImageProcessor
        self.args, self.torch = args, torch
        self.path, self.asset = resolve_asset(args, args.model)
        self.dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
        self.model = None
        self.clip = None
        self.extra = {}
        if args.model == "chexagent8b":
            self.processor = AutoProcessor.from_pretrained(self.path, trust_remote_code=True, local_files_only=True)
            self.tokenizer = self.processor.tokenizer
            if not args.inspect:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.path, trust_remote_code=True, local_files_only=True,
                    torch_dtype=self.dtype, low_cpu_mem_usage=True).to(args.device).eval()
        else:
            sys.path.insert(0, str(args.llava_repo.resolve()))
            from llava.model import LlavaMistralForCausalLM, LlavaMistralConfig
            from llava.conversation import conv_templates
            from llava.mm_utils import tokenizer_image_token, process_images
            self.conv_templates, self.tokenizer_image_token, self.process_images = conv_templates, tokenizer_image_token, process_images
            self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True)
            self.config = LlavaMistralConfig.from_pretrained(self.path, local_files_only=True)
            self.clip, clip_asset = resolve_asset(args, "clip336")
            self.processor = CLIPImageProcessor.from_pretrained(self.clip, local_files_only=True)
            self.extra.update(clip_path=str(self.clip), clip_asset=clip_asset,
                clip_config_sha256=sha(self.clip / "config.json"),
                original_tokenizer_model_max_length=self.config.tokenizer_model_max_length,
                native_inference_context=args.max_context,
                llava_repo_commit=subprocess.check_output(["git", "-C", str(args.llava_repo), "rev-parse", "HEAD"], text=True).strip(),
                llava_source_sha256={str(p.relative_to(args.llava_repo)):sha(p) for p in (args.llava_repo / "llava").rglob("*.py")})
            self.config.mm_vision_tower = str(self.clip)
            # Native prepare_inputs otherwise silently cuts multimodal sequences
            # at a 2,048-token training setting, possibly dropping the question.
            self.config.tokenizer_model_max_length = args.max_context
            if not args.inspect:
                self.model = LlavaMistralForCausalLM.from_pretrained(
                    self.path, config=self.config, local_files_only=True,
                    low_cpu_mem_usage=True, torch_dtype=self.dtype,
                    use_flash_attention_2=False)
                tower = self.model.get_vision_tower()
                if not tower.is_loaded:
                    tower.load_model()
                # Keep precisely the checkpoint vocabulary: this model declares
                # no image patch/start/end vocabulary additions.
                if self.config.mm_use_im_patch_token or self.config.mm_use_im_start_end:
                    raise ValueError("unsupported LLaVA-Med checkpoint image-token contract")
                if len(self.tokenizer) != self.config.vocab_size:
                    raise ValueError("tokenizer/model vocabulary mismatch")
                self.model.to(device=args.device, dtype=self.dtype).eval()
        ids = {s:self.tokenizer.encode(s, add_special_tokens=False) for s in ["Yes", "No"]}
        if any(len(v) != 1 for v in ids.values()):
            raise ValueError(f"protocol requires complete single-token Yes/No: {ids}")
        self.candidates = {s:v[0] for s,v in ids.items()}
        for word, token_id in self.candidates.items():
            if self.tokenizer.decode([token_id]).strip() != word:
                raise ValueError("candidate decode does not match complete word")
        self.metadata = dict(model=args.model, checkpoint=str(self.path), asset=self.asset,
            config_sha256=sha(self.path / "config.json"), candidate_ids=self.candidates,
            dtype=str(self.dtype), transformers=transformers.__version__,
            model_code_sha256={p.name:sha(p) for p in self.path.glob("*.py")},
            model_weight_files={p.name:p.stat().st_size for p in self.path.glob("*.safetensors")},
            template="official USER: <s>... ASSISTANT: <s>" if args.model == "chexagent8b" else "official mistral_instruct",
            probability="full-vocabulary raw log_softmax, then binary Yes/(Yes+No)", **self.extra)
        if self.model is not None:
            self.metadata["parameters"] = sum(p.numel() for p in self.model.parameters())

    def inputs(self, image, prompt):
        torch = self.torch
        if self.args.model == "chexagent8b":
            inputs = self.processor(images=[image], text=f" USER: <s>{prompt} ASSISTANT: <s>", return_tensors="pt")
            if inputs["input_ids"].shape[1] + 128 > self.args.max_context:
                raise ValueError("CheXagent context exceeds the matched limit")
            return {k:v.to(device=self.args.device, dtype=self.dtype if v.is_floating_point() else v.dtype) for k,v in inputs.items()}
        conv = self.conv_templates["mistral_instruct"].copy()
        conv.append_message(conv.roles[0], "<image>\n"+prompt)
        conv.append_message(conv.roles[1], None)
        ids = self.tokenizer_image_token(conv.get_prompt(), self.tokenizer, -200, return_tensors="pt").unsqueeze(0)
        if ids.shape[1] + 576 - 1 > self.args.max_context:
            raise ValueError("LLaVA-Med context exceeds the matched limit; refusing silent crop")
        pixels = self.process_images([image], self.processor, self.config)
        if not isinstance(pixels, torch.Tensor):
            raise ValueError("expected a single stacked LLaVA-Med image")
        return dict(input_ids=ids.to(self.args.device), images=pixels.to(device=self.args.device, dtype=self.dtype))

    def predict(self, image, prompt, probability, max_new_tokens):
        torch = self.torch
        inputs = self.inputs(image, prompt)
        with torch.inference_mode():
            if probability:
                # Forward bypasses every generation logits processor, forced
                # candidate mask, repetition penalty and renormalization.
                logits = self.model(**inputs, return_dict=True).logits[0, -1].float()
                lp = torch.log_softmax(logits, dim=-1)
                selected = {s:float(lp[i]) for s,i in self.candidates.items()}
                candidate_logits = logits[[self.candidates["Yes"], self.candidates["No"]]]
                value = float(torch.softmax(candidate_logits, dim=0)[0])
                if not all(torch.isfinite(torch.tensor(v)) for v in selected.values()):
                    raise ValueError("nonfinite raw candidate log probabilities")
                return dict(probability=value, candidate_logprobs=selected, requests=1,
                            usage=dict(prompt_tokens=inputs["input_ids"].shape[1], completion_tokens=0))
            prompt_tokens = inputs["input_ids"].shape[1]
            kwargs = dict(do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
                          repetition_penalty=1., use_cache=True, temperature=1., top_p=1.,
                          eos_token_id=self.tokenizer.eos_token_id,
                          pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
            if self.args.model == "llava_med7b":
                output = self.model.generate(inputs=inputs.pop("input_ids"), **inputs, **kwargs)[0]
            else:
                output = self.model.generate(**inputs, **kwargs)[0]
            # Both native wrappers generate via inputs_embeds and return only a
            # synthetic BOS plus generated IDs, never the original input IDs.
            text = self.tokenizer.decode(output, skip_special_tokens=True).strip()
            return dict(text=text, finish_reason="stop" if int(output[-1]) == self.tokenizer.eos_token_id else "length",
                        usage=dict(prompt_tokens=prompt_tokens, completion_tokens=len(output)))


def main(args):
    os.umask(0o077)
    sys.path.insert(0, str(args.vendor.resolve()))
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    import torch
    import transformers
    if transformers.__version__ != "4.36.2":
        raise RuntimeError("legacy_vlm requires isolated transformers==4.36.2; shared packages must remain unchanged")
    torch.set_num_threads(args.threads)
    torch.manual_seed(20260911)
    source = args.source or args.run / "source"
    if not source.exists() and args.inspect:
        source = PROJECT / "code/medworld_baselines"
    sys.path.insert(0, str(source.resolve()))
    import infer as protocol
    if Path(protocol.__file__).resolve().parent != source.resolve():
        raise ValueError("wrong baseline prompt module imported")
    out = args.run / args.model / args.split
    out.mkdir(parents=True, exist_ok=True)
    with (out / "native_inference.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        engine = NativeModel(args)
        if args.inspect:
            from PIL import Image
            args.device = "cpu"
            tensors = engine.inputs(Image.new("RGB", (512, 512), 127), "Is edema present? Answer exactly Yes or No.")
            result = dict(status="passed", scope="metadata/processor-only; weights not loaded; no effect estimate",
                          input_shapes={k:list(v.shape) for k,v in tensors.items()}, **engine.metadata)
            atomic(out / "native_inspection.json", result)
            print(json.dumps(result), flush=True)
            return
        if not (args.run / "protocol.json").exists():
            raise ValueError("prepare the identical input protocol/cohort before inference")
        plan = dict(protocol_sha256=sha(args.run / "protocol.json"), checkpoint=str(engine.path),
                    split=args.split, candidates=engine.candidates,
                    prompt_source_sha256=sha(source / "infer.py"),
                    source_sha256=sha(Path(__file__)), max_report_tokens=384, max_qa_tokens=192,
                    max_context=args.max_context, native_model=engine.metadata,
                    input_sha256={name:sha(args.run / "cohort" / f"{name}_inputs_{args.split}.jsonl") for name in ["table1", "table2", "qa"]})
        if (out / "plan.json").exists() and json.loads((out / "plan.json").read_text()) != plan:
            raise ValueError("resuming with a different prompt/model/cohort/native implementation")
        atomic(out / "plan.json", plan)
        vocabulary = protocol.read(args.run / "vocabulary.json")
        table1 = protocol.rows(args.run / "cohort" / f"table1_inputs_{args.split}.jsonl")[:args.limit or None]
        table2 = protocol.rows(args.run / "cohort" / f"table2_inputs_{args.split}.jsonl")[:args.limit or None]
        qa = protocol.rows(args.run / "cohort" / f"qa_inputs_{args.split}.jsonl")[:args.limit or None]
        groups = dict(table1_report=[(r,None) for r in table1],
            table1_prob=[(r,f) for r in table1 for f in protocol.FUTURE_FINDINGS],
            table2_report=[(r,None) for r in table2 if r["report_generation"]],
            table2_prob=[(r,f) for r in table2 if r["classification"] for f in protocol.CURRENT_FINDINGS],
            qa=[(r,None) for r in qa])
        chosen = list(groups) if args.tasks == "all" else args.tasks.split(",")
        if any(t not in groups for t in chosen):
            raise ValueError("unknown task")
        journal = out / "responses.jsonl"
        done = set()
        if journal.exists():
            raw = journal.read_bytes().splitlines(keepends=True)
            if raw and not raw[-1].endswith(b"\n"):
                journal.write_bytes(b"".join(raw[:-1]))
            done = {r["key"] for r in protocol.rows(journal) if r.get("ok")}
        stop = []
        def receive(signum, frame):
            stop.append(signum)
        signal.signal(signal.SIGTERM, receive)
        signal.signal(signal.SIGUSR1, receive)
        state = dict(model=args.model, split=args.split, started=time.time(), pid=os.getpid(), status="running", tasks={},
                     mode="native Transformers", cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                     smoke=bool(args.limit))
        def flush():
            state["updated"] = time.time()
            atomic(out / "status.json", state)
        with journal.open("a", buffering=1) as log:
            from PIL import Image
            for task in chosen:
                items = groups[task]
                state["current_task"] = task
                state["tasks"][task] = dict(expected=len(items), completed=0, errors=0)
                for row, finding in items:
                    key = task + "|" + row["id"] + "|" + (finding or "")
                    if key in done:
                        state["tasks"][task]["completed"] += 1
                        continue
                    if stop:
                        state.update(status="paused", reason=f"signal {stop[-1]}")
                        flush()
                        raise SystemExit(124)
                    started = time.monotonic()
                    result = dict(key=key, task=task, id=row["id"], finding=finding, ok=False)
                    try:
                        # Exactly the same 512px black-padded PNG that all
                        # matched endpoint baselines receive, decoded in memory.
                        encoded = protocol.image_url(row["image"]).split(",",1)[1]
                        image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
                        prompt = protocol.prompt_for(task, row, finding, vocabulary)
                        result.update(engine.predict(image, prompt, task.endswith("_prob"), 192 if task == "qa" else 384), ok=True)
                    except Exception as exc:
                        result["error"] = f"{type(exc).__name__}: {exc}"
                        if isinstance(exc, torch.cuda.OutOfMemoryError):
                            torch.cuda.empty_cache()
                    result.update(seconds=time.monotonic()-started, attempts=1)
                    log.write(json.dumps(result, ensure_ascii=False, allow_nan=False)+"\n")
                    if result["ok"]:
                        done.add(key)
                        state["tasks"][task]["completed"] += 1
                    else:
                        state["tasks"][task]["errors"] += 1
                    flush()
                print(args.model, task, json.dumps(state["tasks"][task]), flush=True)
        state["status"] = "complete" if all(v["expected"] == v["completed"] for v in state["tasks"].values()) else "partial_errors"
        state["finished"] = time.time()
        state["seconds"] = state["finished"] - state["started"]
        flush()
        atomic(args.run / args.model / "inference_finished.json", dict(complete=state["status"] == "complete", **state))
        if state["status"] != "complete":
            raise SystemExit(2)


if __name__ == "__main__":
    main(args_parser())
