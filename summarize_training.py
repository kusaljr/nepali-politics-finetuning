"""Collect compact training results from completed runs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


MAIN_RUNS = (
    "lora-r16-e3-l1024-seed42",
    "lora-r16-e3-l1024-seed7",
    "lora-r16-e3-l1024-seed123",
)


def compact(result: dict) -> dict:
    curve = []
    for row in result["log_history"]:
        if "eval_loss" in row:
            curve.append({
                "step": row["step"],
                "epoch": row["epoch"],
                "eval_loss": row["eval_loss"],
                "token_accuracy": row.get("eval_mean_token_accuracy"),
            })
    return {
        key: result[key]
        for key in (
            "name", "method", "max_length", "assistant_only_loss", "lora_r",
            "epochs", "seed", "split_seed", "train_batch_size",
            "eval_batch_size", "gradient_accumulation_steps", "train_examples",
            "eval_examples", "elapsed_seconds", "peak_gpu_memory_gb",
            "train_metrics", "eval_metrics",
        )
    } | {"eval_curve": curve}


def seed_summary(runs: dict[str, dict]) -> dict | None:
    if not all(name in runs for name in MAIN_RUNS):
        return None
    fields = {
        "eval_loss": [runs[name]["eval_metrics"]["eval_loss"] for name in MAIN_RUNS],
        "token_accuracy": [
            runs[name]["eval_metrics"]["eval_mean_token_accuracy"]
            for name in MAIN_RUNS
        ],
    }
    return {
        field: {
            "values": values,
            "mean": statistics.mean(values),
            "sample_std": statistics.stdev(values),
        }
        for field, values in fields.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, default=Path("results/training_results.json"))
    args = parser.parse_args()

    runs = {}
    for path in sorted(args.runs.glob("*/result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        runs[result["name"]] = compact(result)
    report = {"runs": runs, "three_seed_summary": seed_summary(runs)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(runs)} runs to {args.output}")


if __name__ == "__main__":
    main()
