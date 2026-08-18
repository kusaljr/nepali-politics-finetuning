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

This card documents the reference LoRA run on the synthetic
`nepali_politics_news.jsonl` dataset. Model weights are kept outside the Git
repository; the training configuration and compact results are versioned here.

## Intended use

The adapter is intended for research on Nepali generation, register transfer,
and the limits of supervised fine-tuning in a small language model. It is not a
source of current or verified political information.

## Training data

The dataset contains 3,460 synthetic multi-turn conversations about Nepali
politics and elections. A seeded 90/10 train/test split is created in the
notebook. Assistant-only masking is used in the reference LoRA configuration.

## Evaluation

Evaluation covers the base, instruction-prefix, three-shot, nearest-neighbour,
copy, and fine-tuned systems on the complete held-out split. Metrics include
Unicode-aware and default-tokenizer ROUGE-L, chrF, distinct-2, rep-4,
Nepali/Hindi language identification, and entity precision and recall, with
bootstrap confidence intervals.

## Limitations

- The data are synthetic and may contain factual errors or stylistic artifacts.
- A 270M-parameter model may learn language and register without learning the
  underlying political facts.
- Generated names, dates, numbers, and vote counts require verification.
- The adapter must not be used for election reporting or political decisions.

## Reproduction

Exact dependency versions are in `requirements.txt`. The notebook and command
line scripts reproduce training and evaluation. Final hyperparameters,
seed-level results, and compute details are recorded after the suite completes.
