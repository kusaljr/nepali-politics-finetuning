"""Check that the published result artifacts cover the full study."""

from __future__ import annotations

import json
import math
from pathlib import Path


TRAINING_RUNS = {
    "lora-r16-e3-l1024-seed42",
    "lora-r16-e3-l1024-seed7",
    "lora-r16-e3-l1024-seed123",
    "lora-r16-e3-l512-seed42",
    "lora-r16-e3-l1024-allloss",
    "lora-r8-e3-l1024-seed42",
    "lora-r32-e3-l1024-seed42",
    "lora-r16-e1-l1024-seed42",
    "lora-r16-e5-l1024-seed42",
    "full-e3-l1024-seed42",
}
EVALUATED_SYSTEMS = {
    "base",
    "nepali-instruction",
    "three-shot-nepali",
    "lora-r16-e3-l1024-seed42",
    "lora-r16-e3-l1024-seed7",
    "lora-r16-e3-l1024-seed123",
    "full-e3-l1024-seed42",
    "lora-seed42-sampled",
    "copy-question",
    "char-tfidf-nearest-neighbour",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    training = load(Path("results/training_results.json"))
    metrics = load(Path("results/generation_metrics.json"))
    lengths = load(Path("results/token_lengths.json"))

    assert set(training["runs"]) == TRAINING_RUNS
    assert training["three_seed_summary"] is not None
    for run in training["runs"].values():
        assert run["train_examples"] == 3114
        assert run["eval_examples"] == 346
        assert math.isfinite(run["train_metrics"]["train_loss"])
        assert math.isfinite(run["eval_metrics"]["eval_loss"])
        assert run["common_eval"]["max_length"] == 1024
        assert run["common_eval"]["assistant_only_loss"] is True
        assert run["common_eval"]["eval_examples"] == 346
        assert math.isfinite(run["common_eval"]["metrics"]["eval_loss"])

    assert set(metrics) == EVALUATED_SYSTEMS
    for result in metrics.values():
        assert result["n"] == 1718
        for name in ("rougeL_default", "rougeL_unicode", "chrf", "distinct_2", "rep_4"):
            assert math.isfinite(result[name]["mean"])
            assert len(result[name]["ci95"]) == 2
        assert "language_ne_accuracy" in result
        assert "language_hi_rate" in result
        assert set(result["entities"]) == {"person", "party", "date", "number"}

    assert lengths["conversations"] == 3460
    assert lengths["over_512_count"] == 3221
    assert lengths["over_1024_count"] == 0
    print("result artifacts cover 10 training runs and 10 full-test systems")


if __name__ == "__main__":
    main()
