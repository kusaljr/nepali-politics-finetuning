"""Evaluate a saved run with the common assistant-only loss protocol."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from train_experiment import GEMMA_TEMPLATE, MODEL_NAME, load_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--dataset", type=Path, default=Path("nepali_politics_news.jsonl"))
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    result_path = args.run_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    model_dir = args.run_dir / "model"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.chat_template = GEMMA_TEMPLATE
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    if result["method"] == "lora":
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            token=os.environ.get("HF_TOKEN"),
            dtype=dtype,
            attn_implementation="eager",
        )
        model = PeftModel.from_pretrained(base, model_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            dtype=dtype,
            attn_implementation="eager",
        )
    model.config.use_cache = False
    data = load_data(args.dataset, result["split_seed"], smoke=args.smoke)
    config = SFTConfig(
        output_dir=str(args.run_dir / "common_eval_tmp"),
        max_length=args.max_length,
        packing=False,
        assistant_only_loss=True,
        per_device_eval_batch_size=1,
        eval_strategy="no",
        save_strategy="no",
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=data["train"],
        eval_dataset=data["test"],
        processing_class=tokenizer,
    )
    metrics = trainer.evaluate()
    report = {
        "run": result["name"],
        "max_length": args.max_length,
        "assistant_only_loss": True,
        "eval_examples": len(data["test"]),
        "metrics": metrics,
    }
    output = args.run_dir / ("common_eval_smoke.json" if args.smoke else "common_eval.json")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
