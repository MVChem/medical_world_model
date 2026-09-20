"""Freeze a literature-guided short-answer rerun of the identical pilot."""

import argparse
import copy
import os
import shutil
import subprocess
import time
from pathlib import Path

from .common import atomic, digest, rows, verify, write_rows


def main(previous, run, repositories, prompt_style):
    os.umask(0o077)
    old = verify(previous)
    if run.exists():
        raise FileExistsError(run)
    run.mkdir(parents=True)
    for filename in (
        "references.jsonl",
        "models.json",
        "vocabulary.json",
        "official_ans2idx.json",
        "training_overlap.json",
        "official_split_audit.json",
        "image_sources.json",
    ):
        shutil.copy2(previous / filename, run / filename)
    shutil.copytree(previous / "images", run / "images", copy_function=os.link)
    inputs = rows(previous / "inputs.jsonl")
    for row in inputs:
        row["image"] = str(run / "images" / Path(row["image"]).name)
    write_rows(run / "inputs.jsonl", inputs)
    sources = {}
    requested = {
        "CheXagent": [
            "data_chexinstruct/dataset_processors/templates.py",
            "data_chexinstruct/dataset_processors/mimic_cxr_vqa.py",
            "LICENSE",
        ],
        "AOR": ["README.md"],
        "LLaVA": ["docs/Evaluation.md"],
        "Meissa": [
            "environments/continuous_tool_calling/eval/run_mimic_cxr_vqa.py",
            "LICENSE",
        ],
    }
    for repo, filenames in requested.items():
        root = repositories / repo
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        sources[repo] = {"commit": commit, "files": {}}
        for filename in filenames:
            target = run / "literature" / repo / Path(filename).name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / filename, target)
            sources[repo]["files"][filename] = digest(target)
    sources["paper_metrics"] = "https://arxiv.org/html/2505.02830v1#S5"
    sources["scope"] = (
        "CheXagent open_ended_qa_task has q=question. Final rerun uses the exact short-answer suffix from LLaVA official Evaluation.md. "
        "Unlike its data processor, retain empty answers and global-abnormality questions. "
        "AOR metric definitions only; complete MIMIC evaluator not found in pinned public repo. "
        "Meissa normalize/check_correct executed source-exact, without the agent, for sensitivity only."
    )
    atomic(run / "literature_sources.json", sources)
    package = Path(__file__).parent
    source = run / "source/medworld_vqa"
    shutil.copytree(
        package, source, ignore=shutil.ignore_patterns("runs", "__pycache__")
    )
    shutil.copy2(previous / "source/medworld_vqa/gpu.py", source / "gpu.py")
    atomic(
        run / "source_manifest.json",
        {
            str(p.relative_to(run / "source")): digest(p)
            for p in source.rglob("*")
            if p.is_file()
        },
    )
    protocol = copy.deepcopy(old)
    for key in ("prior_protocol", "amendments", "file_sha256"):
        protocol.pop(key, None)
    protocol.update(
        version="cxrvqa_short_answer_literature_v1",
        created_unix=time.time(),
        prompt_style=prompt_style,
        input="One current image plus question and LLaVA official short-answer instruction; no answer vocabulary",
        generation={
            "temperature": 0,
            "max_tokens": 512,
            "qwen_thinking": False,
            "seed": old["seed"],
            "structured_outputs": None,
        },
        scoring="AOR type metrics, conservative local canonical-text-list parser; source-exact Meissa any-substring sensitivity metric. Unknown fragments add an invalid FP; known canonical list items retain partial credit. Truncation/missing response invalidates all labels.",
        scope="Same 1024-question sampled zero-shot rerun, no task fine-tuning; published prompt and metric definitions, NOT a full reproduction of supervised paper experiments",
        prior_run=str(previous),
        prior_protocol_sha256=digest(previous / "protocol.json"),
    )
    protocol["file_sha256"] = {
        str(p.relative_to(run)): digest(p)
        for p in run.rglob("*")
        if p.is_file() and "source" not in p.relative_to(run).parts
    }
    atomic(run / "protocol.json", protocol)
    verify(run)
    print(f"Frozen {len(inputs)} identical QA inputs: {run}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--repositories", type=Path, default=Path("/tmp/cxrvqa_published_research")
    )
    args = parser.parse_args()
    main(
        args.previous.resolve(),
        args.run.resolve(),
        args.repositories.resolve(),
        "llava_short_answer",
    )
