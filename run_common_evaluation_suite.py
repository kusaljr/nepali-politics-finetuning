"""Rescore completed checkpoints with the common loss protocol."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from run_training_suite import EXPERIMENTS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    args = parser.parse_args()

    for experiment in EXPERIMENTS:
        run_dir = args.runs / experiment["name"]
        if (run_dir / "common_eval.json").exists():
            print(f"skip {experiment['name']}", flush=True)
            continue
        if not (run_dir / "result.json").exists():
            raise FileNotFoundError(f"missing result for {experiment['name']}")
        print(f"evaluate {experiment['name']}", flush=True)
        subprocess.run(
            [sys.executable, "evaluate_training_loss.py", str(run_dir)],
            check=True,
        )


if __name__ == "__main__":
    main()
