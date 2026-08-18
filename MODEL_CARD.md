---
language:
- ne
base_model: google/gemma-3-270m-it
library_name: peft
pipeline_tag: text-generation
license: gemma
tags:
- lora
- nepali
- question-answering
---

# Gemma-3-270M Nepali Election Q&A Adapter

This model card accompanies a LoRA adapter trained on the synthetic
`nepali_politics_news.jsonl` dataset. The adapter artifact has not yet been
published from this repository; reported metadata must be completed from the
final reproducible run before release.

## Intended use

The adapter is intended for research on Nepali generation, register transfer,
and the limits of supervised fine-tuning in a small language model. It is not a
source of current or verified political information.

## Training data

The dataset contains 3,460 synthetic multi-turn conversations about Nepali
politics and elections. A seeded 90/10 train/test split is created in the
notebook. Assistant-only masking is used in the reference LoRA configuration.

## Evaluation

The final release should report results for the base, system-prompt, three-shot,
nearest-neighbour, copy, and fine-tuned systems on the complete held-out split.
Metrics should include Unicode-aware and default-tokenizer ROUGE-L, chrF,
distinct-2, rep-4, Nepali/Hindi language identification, and entity precision
and recall, with bootstrap confidence intervals.

## Limitations

- The data are synthetic and may contain factual errors or stylistic artifacts.
- A 270M-parameter model may learn language and register without learning the
  underlying political facts.
- Generated names, dates, numbers, and vote counts require verification.
- The adapter must not be used for election reporting or political decisions.

## Reproduction

Exact dependency versions are in `requirements.txt`; training and evaluation
code are in `finetune_nepali.ipynb`. Add the final hyperparameters, seed-level
results, compute details, and adapter repository identifier before publishing.
