"""Run one reproducible Gemma fine-tuning experiment."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


MODEL_NAME = "google/gemma-3-270m-it"
GEMMA_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "{{ '<start_of_turn>user\n' + message['content'] | trim + '<end_of_turn>\n' }}"
    "{% elif message['role'] == 'assistant' or message['role'] == 'model' %}"
    "{{ '<start_of_turn>model\n' }}"
    "{% generation %}{{ message['content'] | trim + '<end_of_turn>\n' }}{% endgeneration %}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<start_of_turn>model\n' }}{% endif %}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--dataset", type=Path, default=Path("nepali_politics_news.jsonl"))
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--method", choices=("lora", "full"), default="lora")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--assistant-only-loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--train-batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data(path: Path, split_seed: int, smoke: bool):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            messages = json.loads(line)
            rows.append({"messages": messages})
    split = Dataset.from_list(rows).train_test_split(test_size=0.1, seed=split_seed)
    if smoke:
        split["train"] = split["train"].select(range(min(32, len(split["train"]))))
        split["test"] = split["test"].select(range(min(16, len(split["test"]))))
    return split


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    set_seed(args.seed)
    output_dir = args.output_root / args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    is_lora = args.method == "lora"
    train_batch = args.train_batch_size or (8 if is_lora else 2)
    eval_batch = args.eval_batch_size or (8 if is_lora else 2)
    accumulation = args.gradient_accumulation_steps or (2 if is_lora else 8)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=os.environ.get("HF_TOKEN"))
    tokenizer.chat_template = GEMMA_TEMPLATE
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=os.environ.get("HF_TOKEN"),
        dtype=torch.float16,
        attn_implementation="eager",
    )
    model.config.use_cache = False
    data = load_data(args.dataset, args.split_seed, args.smoke)

    rendered = tokenizer.apply_chat_template(
        data["train"][0]["messages"], tokenize=True, return_dict=True
    )["input_ids"]
    leading_bos = 0
    for token_id in rendered:
        if token_id != tokenizer.bos_token_id:
            break
        leading_bos += 1
    if leading_bos != 1:
        raise RuntimeError(f"expected one BOS token, found {leading_bos}")

    peft_config = None
    if is_lora:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=32,
            lora_dropout=0.05,
            use_rslora=True,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )

    config = SFTConfig(
        output_dir=str(output_dir / "checkpoints"),
        max_length=args.max_length,
        packing=False,
        assistant_only_loss=args.assistant_only_loss,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=train_batch,
        per_device_eval_batch_size=eval_batch,
        gradient_accumulation_steps=accumulation,
        learning_rate=2e-4 if is_lora else 5e-5,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=20,
        optim="adamw_torch_fused",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="no",
        gradient_checkpointing=True,
        group_by_length=True,
        seed=args.seed,
        data_seed=args.seed,
        fp16=True,
        bf16=False,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=data["train"],
        eval_dataset=data["test"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    started = time.time()
    train_result = trainer.train()
    eval_metrics = trainer.evaluate()
    elapsed = time.time() - started
    model_dir = output_dir / "model"
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(model_dir)

    result = {
        "name": args.name,
        "method": args.method,
        "max_length": args.max_length,
        "assistant_only_loss": args.assistant_only_loss,
        "lora_r": args.lora_r if is_lora else None,
        "epochs": args.epochs,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "train_batch_size": train_batch,
        "eval_batch_size": eval_batch,
        "gradient_accumulation_steps": accumulation,
        "train_examples": len(data["train"]),
        "eval_examples": len(data["test"]),
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
        "log_history": trainer.state.log_history,
    }
    with (output_dir / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "log_history"}, indent=2))


if __name__ == "__main__":
    main()
