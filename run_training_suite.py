"""Run the non-duplicated training matrix sequentially."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


EXPERIMENTS = [
    {"name": "lora-r16-e3-l1024-seed42"},
    {"name": "lora-r16-e3-l1024-seed7", "seed": 7},
    {"name": "lora-r16-e3-l1024-seed123", "seed": 123},
    {"name": "lora-r16-e3-l512-seed42", "max_length": 512},
    {"name": "lora-r16-e3-l1024-allloss", "assistant_only_loss": False},
    {"name": "lora-r8-e3-l1024-seed42", "lora_r": 8},
    {"name": "lora-r32-e3-l1024-seed42", "lora_r": 32},
    {"name": "lora-r16-e1-l1024-seed42", "epochs": 1},
    {"name": "lora-r16-e5-l1024-seed42", "epochs": 5},
    {"name": "full-e3-l1024-seed42", "method": "full"},
]


def command(experiment: dict, output_root: Path, smoke: bool) -> list[str]:
    values = {
        "method": "lora",
        "max_length": 1024,
        "lora_r": 16,
        "epochs": 3,
        "seed": 42,
        **experiment,
    }
    cmd = [
        sys.executable, "train_experiment.py",
        "--name", values["name"],
        "--output-root", str(output_root),
        "--method", values["method"],
        "--max-length", str(values["max_length"]),
        "--lora-r", str(values["lora_r"]),
        "--epochs", str(values["epochs"]),
        "--seed", str(values["seed"]),
    ]
    if values.get("assistant_only_loss", True):
        cmd.append("--assistant-only-loss")
    else:
        cmd.append("--no-assistant-only-loss")
    if smoke:
        cmd.append("--smoke")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    selected = [e for e in EXPERIMENTS if not args.only or e["name"] in args.only]
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for experiment in selected:
        result_path = args.output_root / experiment["name"] / "result.json"
        if result_path.exists() and not args.smoke:
            print(f"skip completed {experiment['name']}", flush=True)
            continue
        started = time.time()
        print(f"start {experiment['name']}", flush=True)
        completed = subprocess.run(command(experiment, args.output_root, args.smoke))
        record = {
            "name": experiment["name"],
            "returncode": completed.returncode,
            "wall_seconds": time.time() - started,
        }
        manifest.append(record)
        with (args.output_root / "suite_manifest.json").open("w") as handle:
            json.dump(manifest, handle, indent=2)
        if completed.returncode:
            raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
