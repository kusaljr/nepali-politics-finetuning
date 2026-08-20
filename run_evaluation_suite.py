"""Generate all held-out predictions and produce the final metric report."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path


MAIN_RUNS = (
    "lora-r16-e3-l1024-seed42",
    "lora-r16-e3-l1024-seed7",
    "lora-r16-e3-l1024-seed123",
)


def run(command: list[str]) -> None:
    print("run", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, default=Path("evaluation"))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    base_path = args.output / "base_predictions.jsonl"
    if not base_path.exists():
        run([
            sys.executable, "generate_predictions.py",
            "--output", str(base_path),
            "--batch-size", str(args.batch_size),
        ])

    prediction_paths = [base_path]
    for run_name in MAIN_RUNS:
        output_path = args.output / f"{run_name}.jsonl"
        prediction_paths.append(output_path)
        if not output_path.exists():
            run([
                sys.executable, "generate_predictions.py",
                "--skip-base",
                "--adapter", str(args.runs / run_name / "model"),
                "--fine-tuned-name", run_name,
                "--output", str(output_path),
                "--batch-size", str(args.batch_size),
            ])

    full_path = args.output / "full-e3-l1024-seed42.jsonl"
    prediction_paths.append(full_path)
    if not full_path.exists():
        run([
            sys.executable, "generate_predictions.py",
            "--skip-base",
            "--fine-tuned-model", str(args.runs / "full-e3-l1024-seed42" / "model"),
            "--fine-tuned-name", "full-e3-l1024-seed42",
            "--output", str(full_path),
            "--batch-size", str(args.batch_size),
        ])

    sampled_path = args.output / "lora-seed42-sampled.jsonl"
    prediction_paths.append(sampled_path)
    if not sampled_path.exists():
        run([
            sys.executable, "generate_predictions.py",
            "--skip-base", "--sample",
            "--adapter", str(args.runs / MAIN_RUNS[0] / "model"),
            "--fine-tuned-name", "lora-seed42-sampled",
            "--output", str(sampled_path),
            "--batch-size", str(args.batch_size),
        ])

    text_path = args.output / "text_baselines.jsonl"
    if not text_path.exists():
        run([sys.executable, "evaluate.py", "--build-text-baselines", "--output", str(text_path)])
    prediction_paths.append(text_path)

    combined_path = args.output / "all_predictions.jsonl"
    with combined_path.open("w", encoding="utf-8") as output:
        for path in prediction_paths:
            with path.open(encoding="utf-8") as source:
                for line in source:
                    output.write(line)

    lid_path = args.output / "lid.176.bin"
    if not lid_path.exists():
        urllib.request.urlretrieve(
            "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
            lid_path,
        )
    report = subprocess.run(
        [
            sys.executable, "evaluate.py", str(combined_path),
            "--lid-model", str(lid_path),
            # CIs for full fine-tune and the seed-7 LoRA run overlap heavily;
            # this settles which (if either) actually wins on chrF.
            "--compare", "full-e3-l1024-seed42", "lora-r16-e3-l1024-seed7",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    metrics = json.loads(report.stdout)
    with (args.output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    results_path = Path("results/generation_metrics.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    print(report.stdout)


if __name__ == "__main__":
    main()
