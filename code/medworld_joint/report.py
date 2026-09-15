"""Aggregate completed online adaptation pilots without mixing baseline protocols."""
import argparse
import json
from pathlib import Path


def report(run):
    rows = []
    for path in sorted(run.glob("*/*/metrics.json")):
        if not path.with_name("complete.json").exists():
            continue
        item = json.loads(path.read_text())
        contract = json.loads(path.with_name("contract.json").read_text())
        test = item["metrics"].get("test", {})
        human = item["metrics"].get("human_test", {})
        rows.append(dict(model=item["model"], task=item["task"], condition=item["condition"],
            steps=item.get("training_steps"), epochs=item.get("training_epoch_fraction"),
            train_n=item["train_n"], eval_limit=contract["eval_limit"],
            pseudo_dice=test.get("dice"), human_lung_dice=human.get("dice"),
            psnr=test.get("psnr"), ssim=test.get("ssim"),
            metrics_path=str(path.resolve()), pilot=True, world_model_pretraining=False))
    (run / "report.json").write_text(json.dumps(dict(results=rows), indent=2) + "\n")
    lines = ["# Online downstream adaptation pilot", "",
        "Original pretrained VLM initialization; no longitudinal/world-model pretraining. "
        "Eight-slot conditions use four language depths and four native vision depths; "
        "both dense tasks read all eight. Segmentation test Dice is CXAS teacher agreement, "
        "with Montgomery human lung Dice reported separately. SR inputs are LR only.", "",
        "| Model | Task | Condition | Steps | Train n | Test Dice | Human lung Dice | PSNR | SSIM |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    def cell(value):
        return "—" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value)
    for row in rows:
        lines.append("| " + " | ".join(cell(row[key]) for key in
            ("model", "task", "condition", "steps", "train_n", "pseudo_dice", "human_lung_dice", "psnr", "ssim")) + " |")
    (run / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(completed=len(rows), report=str(run / "REPORT.md"))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    report(parser.parse_args().run)
