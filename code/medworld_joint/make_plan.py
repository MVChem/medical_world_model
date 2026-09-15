"""Create the bounded overnight scheduler manifest; this command launches nothing."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil


def make_plan(args):
    project, run, source, data = map(lambda path: path.resolve(),
                                    (args.project, args.run, args.source, args.data_run))
    run.mkdir(parents=True, exist_ok=True)
    jobs = []
    microbatches = {"qwen08b": args.qwen08_microbatch, "qwen4b": args.qwen4_microbatch,
                    "medgemma4b": args.medgemma4_microbatch}
    # Actual native GPU smokes: Qwen08 1.33s/32, Qwen4 2.18s/32,
    # MedGemma4 6.43s/32 (two microbatches16). Per-job margin is added below.
    seconds_per_image = {"qwen08b": .042, "qwen4b": .070, "medgemma4b": .202}
    files = ["medworld_joint/" + name for name in
             ("__init__.py", "model.py", "train.py", "report.py", "make_plan.py", "test_joint.py", "README.md")]
    files += ["medworld_dense_baselines/" + name for name in
              ("common.py", "heads.py", "frozen_slots_extract.py", "frozen_slots_train.py")]
    source_hashes = {}
    for name in files:
        origin, target = project / "code" / name, source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        raw = origin.read_bytes()
        if target.exists() and target.read_bytes() != raw:
            raise ValueError(f"immutable source snapshot differs: {target}")
        if not target.exists():
            shutil.copyfile(origin, target)
        source_hashes[name] = hashlib.sha256(raw).hexdigest()
    (run / "source_manifest.json").write_text(json.dumps(dict(source=str(source), sha256=source_hashes), indent=2) + "\n")
    models = args.models.split(",")
    manifest = json.loads((data / "data/manifest.json").read_text())
    train_n = manifest["coverage"]["images"]["train"]
    conditions = ["joint_slots", "joint_full_tokens", "shuffled_slots", "frozen_slots"]
    for model_index, mid in enumerate(models):
        for condition_index, condition in enumerate(conditions):
            for task in ("segmentation", "sr"):
                out = run / mid / f"{task}_{condition}"
                argv = [args.python, "-u", str(source / "medworld_joint/train.py"),
                    "--data-run", str(data), "--out", str(out), "--model", mid,
                    "--task", task, "--condition", condition, "--epochs", str(args.epochs),
                    "--max-steps", "0", "--batch-size", str(args.batch_size),
                    "--microbatch", str(microbatches[mid]), "--checkpoint-every", "50",
                    "--validate-every", "0", "--deadline", args.deadline]
                artifacts = [str(out / name) for name in ("complete.json", "metrics.json", "gradient_audit.json")]
                estimate = train_n * args.epochs * seconds_per_image[mid]
                estimate *= .70 if condition == "frozen_slots" else 1.
                jobs.append(dict(id=f"joint_{mid}_{task}_{condition}", argv=argv, artifacts=artifacts,
                    deps=[], gpus=1, allowed_gpus=args.gpus, priority=10 + model_index * 20 + condition_index * 2,
                    minimum_seconds=math.ceil(estimate * 1.05 + 180),
                    env=dict(MEDWORLD_PROJECT=str(project), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"),
                    success_fields={str(out / "complete.json"): {"complete": True, "epochs_completed": args.epochs},
                                    str(out / "metrics.json"): {"train_n": train_n, "training_epoch_fraction": args.epochs}}))
    for task in ("segmentation", "sr"):
        out = run / "shared" / f"{task}_image_only"
        jobs.append(dict(id=f"joint_shared_{task}_image_only", argv=[args.python, "-u",
            str(source / "medworld_joint/train.py"), "--data-run", str(data), "--out", str(out),
            "--model", "qwen08b", "--task", task, "--condition", "image_only", "--epochs", str(args.epochs),
            "--max-steps", "0", "--batch-size", str(args.batch_size), "--microbatch", "8",
            "--validate-every", "0", "--deadline", args.deadline],
            artifacts=[str(out / name) for name in ("complete.json", "metrics.json", "gradient_audit.json")],
            success_fields={str(out / "complete.json"): {"complete": True, "epochs_completed": args.epochs}},
            deps=[], gpus=1, allowed_gpus=args.gpus, priority=5, minimum_seconds=600,
            env=dict(MEDWORLD_PROJECT=str(project))))
    plan = dict(version=1, protocol="joint native VLM pilot; 4+4 multidepth; dense all8 routing",
        epochs=args.epochs, batch_size=args.batch_size, train_n=train_n, jobs=jobs,
        source_sha256=source_hashes,
        report_commands=[[args.python, str(source / "medworld_joint/report.py"), "--run", str(run)]])
    plan_path = run / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("immutable plan differs")
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    labels = {"qwen08b": "Qwen3.5-0.8B", "qwen4b": "Qwen3.5-4B", "medgemma4b": "MedGemma-1.5-4B"}
    exports = []
    for mid in models + ["shared"]:
        for condition in (["image_only"] if mid == "shared" else conditions):
            name = "Shared image-only decoder" if mid == "shared" else labels[mid] + " / " + condition
            exports.append(dict(table=2, method=name,
                protocol=f"J1: {train_n} train; {args.epochs} epochs; 4+4 native multidepth; all8 routing; downstream pilot",
                artifacts=[dict(path=str(run / mid / f"segmentation_{condition}/metrics.json"),
                    metrics={"Dice pseudo": "metrics.test.dice", "Dice human": "metrics.human_test.dice"}),
                    dict(path=str(run / mid / f"sr_{condition}/metrics.json"),
                    metrics={"PSNR x4": "metrics.test.psnr", "SSIM x4": "metrics.test.ssim"})]))
    (run / "table_export.json").write_text(json.dumps(exports, indent=2) + "\n")
    print(json.dumps(dict(path=str(run / "plan.json"), jobs=len(jobs),
                         estimated_gpu_hours=sum(job["minimum_seconds"] for job in jobs) / 3600)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data-run", type=Path, required=True)
    parser.add_argument("--python", default="/home/data2/chk/workspace/2026/.venv/bin/python")
    parser.add_argument("--models", default="qwen08b,qwen4b,medgemma4b")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--qwen08-microbatch", type=int, default=32)
    parser.add_argument("--qwen4-microbatch", type=int, default=32)
    parser.add_argument("--medgemma4-microbatch", type=int, default=16)
    parser.add_argument("--gpus", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--deadline", default="2026-09-14T07:45:00+08:00")
    make_plan(parser.parse_args())
