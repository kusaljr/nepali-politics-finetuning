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

On all 1,718 held-out turns, the reference LoRA run obtains chrF 38.07,
Unicode-aware ROUGE-L 0.229, 99.6% Nepali language identification, and entity
precision/recall of 0.234/0.215. The base model obtains chrF 17.09 and 72.4%
Nepali identification. A full fine-tune obtains chrF 39.41 and ROUGE-L 0.238.
The three LoRA seeds score 38.07, 39.44, and 38.26 chrF — the full fine-tune
and the seed-7 LoRA run are not distinguishable on chrF once the comparison
accounts for turns being clustered by conversation (see README).

A zero-training character TF-IDF nearest-neighbour baseline scores 32.73
chrF, so fine-tuning's gain over a lookup table is 6.7 chrF, not the 22-point
gain visible against the base model. Fine-tuning's clearer advantage is
entity recall (0.215 vs. 0.127 for that baseline). The person/date entity
extractor is a regex heuristic audited separately in
`results/entity_extractor_audit.md`: on hand-labeled gold text it gets 0.19
precision / 0.55 recall against real person mentions, which bounds how the
entity numbers above should be read — they are not a clean measurement of
model factuality. Language-ID (fastText `lid.176`) is also unreliable on
short Devanagari text: the copy-question baseline, which echoes the Nepali
question verbatim, scores only 69.9% Nepali, well below what verbatim Nepali
text should score.

Default-tokenizer ROUGE-L is approximately zero for every system; the custom
Unicode tokenizer is required for meaningful Devanagari token overlap. Full
metrics and bootstrap intervals are in `results/generation_metrics.json`,
including the corrected cluster bootstrap, entity extraction, gold-answer
language-ID reference, and paired full-vs-LoRA comparison.

## Limitations

- The data are synthetic and may contain factual errors or stylistic artifacts.
- A 270M-parameter model may learn language and register without learning the
  underlying political facts.
- Generated names, dates, numbers, and vote counts require verification.
- The adapter must not be used for election reporting or political decisions.
- The random 90/10 split is not source-disjoint: a zero-training retrieval
  baseline scores 32.73 chrF, indicating meaningful content overlap between
  train and test questions.

## Reproduction

Exact dependency versions are in `requirements.txt`. The notebook and command
line scripts reproduce training and evaluation. Training ablations and
seed-level results are in `results/training_results.json`. Model weights are not
published by this repository.
