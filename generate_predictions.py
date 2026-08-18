"""Generate turn-level outputs for the base, prompted, few-shot, and SFT systems."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline


NEPALI_INSTRUCTION = "नेपाली भाषामा मात्र समाचार शैलीमा उत्तर दिनुहोस्।"
ACKNOWLEDGEMENT = "हुन्छ।"


def load_split(path: Path, seed: int):
    with path.open(encoding="utf-8") as handle:
        rows = [{"messages": json.loads(line)} for line in handle if line.strip()]
    return Dataset.from_list(rows).train_test_split(test_size=0.1, seed=seed)


def demonstrations(train_split, count: int = 3) -> list[dict]:
    demos = []
    for row in train_split:
        messages = row["messages"]
        for index in range(len(messages) - 1):
            if messages[index]["role"] == "user" and messages[index + 1]["role"] != "user":
                demos.extend([messages[index], messages[index + 1]])
                if len(demos) == count * 2:
                    return demos
    raise ValueError("not enough question-answer pairs for demonstrations")


def turn_tasks(test_split) -> list[dict]:
    tasks = []
    for conversation_id, row in enumerate(test_split):
        history = []
        pending = None
        for turn_id, message in enumerate(row["messages"]):
            if message["role"] == "user":
                history.append(message)
                pending = {
                    "conversation_id": conversation_id,
                    "turn_id": turn_id,
                    "question": message["content"],
                    "history": list(history),
                }
            else:
                if pending is not None:
                    pending["gold"] = message["content"]
                    tasks.append(pending)
                    pending = None
                history.append(message)
    return tasks


def generate_system(pipe, tasks, system_name, prefix, args):
    chats = [prefix + task["history"] for task in tasks]
    decoding = {"do_sample": False}
    if args.sample:
        decoding = {"do_sample": True, "temperature": 0.7, "top_p": 0.9}
    config = GenerationConfig(max_new_tokens=args.max_new_tokens, **decoding)
    outputs = pipe(
        chats,
        generation_config=config,
        batch_size=args.batch_size,
    )
    rows = []
    for task, output in zip(tasks, outputs):
        generated = output[0]["generated_text"]
        prediction = generated[-1]["content"] if isinstance(generated, list) else str(generated)
        token_count = len(pipe.tokenizer(prediction, add_special_tokens=False)["input_ids"])
        rows.append({
            "system": system_name,
            "conversation_id": task["conversation_id"],
            "turn_id": task["turn_id"],
            "question": task["question"],
            "prediction": prediction,
            "gold": task["gold"],
            "generated_tokens": token_count,
            "hit_max_new_tokens": token_count >= args.max_new_tokens,
            "decoding": "sampled" if args.sample else "greedy",
        })
    return rows


def make_pipeline(model_name: str, adapter: Path | None = None, fine_tuned_model: Path | None = None):
    source = str(fine_tuned_model) if fine_tuned_model else model_name
    tokenizer_name = str(adapter) if adapter else source
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForCausalLM.from_pretrained(
        source, dtype=torch.float16, device_map="auto", attn_implementation="eager"
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    return pipeline("text-generation", model=model, tokenizer=tokenizer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("nepali_politics_news.jsonl"))
    parser.add_argument("--model", default="google/gemma-3-270m-it")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--fine-tuned-model", type=Path)
    parser.add_argument("--fine-tuned-name", default="fine-tuned")
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("predictions.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    split = load_split(args.dataset, args.seed)
    tasks = turn_tasks(split["test"])
    demos = demonstrations(split["train"])
    rows = []
    base_pipe = None
    if not args.skip_base:
        base_pipe = make_pipeline(args.model)
        systems = [
            ("base", []),
            ("nepali-instruction", [
                {"role": "user", "content": NEPALI_INSTRUCTION},
                {"role": "model", "content": ACKNOWLEDGEMENT},
            ]),
            ("three-shot-nepali", demos),
        ]
        for name, prefix in systems:
            rows.extend(generate_system(base_pipe, tasks, name, prefix, args))

    if args.adapter:
        if base_pipe is not None:
            del base_pipe
        torch.cuda.empty_cache()
        fine_tuned_pipe = make_pipeline(args.model, args.adapter)
        rows.extend(generate_system(fine_tuned_pipe, tasks, args.fine_tuned_name, [], args))
    elif args.fine_tuned_model:
        if base_pipe is not None:
            del base_pipe
        torch.cuda.empty_cache()
        fine_tuned_pipe = make_pipeline(args.model, fine_tuned_model=args.fine_tuned_model)
        rows.extend(generate_system(fine_tuned_pipe, tasks, args.fine_tuned_name, [], args))

    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
