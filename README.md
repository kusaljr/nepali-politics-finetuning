# Fine-tuning Gemma for Nepali Election Question Answering

This project studies supervised fine-tuning of `google/gemma-3-270m-it` on a
synthetic collection of multi-turn Nepali election question-answer
conversations. The main question is whether small-model fine-tuning improves
Nepali language and news-register control, and whether any improvement extends
to factual entities.

## Dataset

`nepali_politics_news.jsonl` contains 3,460 conversations. Each line is a JSON
array of alternating user and model messages. The notebook creates a seeded
90/10 split (3,114 training conversations and 346 test conversations).

The dataset is synthetic. Results therefore characterize behavior on this
dataset and should not be interpreted as factual coverage of Nepali politics.

## Method

The notebook includes:

- completion-only loss using assistant-span masks;
- LoRA fine-tuning and a switch for full-parameter fine-tuning;
- token-length measurement over `enc["input_ids"]`;
- turn-by-turn evaluation with reference conversation history;
- base, prompted, few-shot, nearest-neighbour, and copy baselines;
- Unicode-aware ROUGE-L, chrF, repetition, diversity, language-ID, and
  entity-overlap metrics.

## Reproducing the experiments

Install the pinned dependencies:

```bash
pip install -r requirements.txt
```

Accept the Gemma license on Hugging Face and make `HF_TOKEN` available in the
environment, then run `finetune_nepali.ipynb` from top to bottom.

Using the Gemma tokenizer, conversation lengths range from 113 to 961 tokens
(median 621, p95 755, p99 815). Of the 3,460 conversations, 3,221 (93.1%) exceed
512 tokens and none exceeds 1,024. A 1,024-token context therefore prevents
training-time truncation in this dataset. Despite this, the 512-token run had
lower common-protocol loss than the 1,024-token run (1.8175 versus 1.8945) and
finished about 7% faster. More retained context did not help this setup.

## Evaluation protocol

The primary comparison uses every held-out conversation. Generated turns are
evaluated against the reference answer while both generative models receive the
same reference history. Greedy decoding is used for the reproducible headline
result; sampled decoding is reported separately when testing repetition.

All systems were evaluated on 1,718 held-out turns. Confidence intervals and
category-level entity counts are in `results/generation_metrics.json`.

| System | chrF | Unicode ROUGE-L | Nepali | Entity P/R | rep-4 |
|---|---:|---:|---:|---:|---:|
| Base | 17.09 | 0.131 | 72.4% | 0.324 / 0.093 | 0.065 |
| Nepali instruction | 15.96 | 0.126 | 73.1% | 0.318 / 0.088 | 0.054 |
| Three-shot Nepali | 16.69 | 0.126 | 82.1% | 0.260 / 0.086 | 0.068 |
| Character TF-IDF retrieval | 32.73 | 0.158 | 100.0% | 0.126 / 0.127 | 0.000 |
| Copy question | 8.50 | 0.118 | 69.9% | 0.524 / 0.061 | 0.000 |
| LoRA, seed 42 | 38.07 | 0.229 | 99.6% | 0.234 / 0.215 | 0.037 |
| Full fine-tune | **39.41** | **0.238** | **100.0%** | 0.247 / 0.225 | 0.029 |
| LoRA, sampled | 37.20 | 0.215 | 99.5% | 0.208 / 0.206 | **0.011** |

Fine-tuning produces a large language-control gain and improves overlap with
the references. It does not establish factual knowledge: entity precision is
only 0.247 for the best system, and the dataset itself is synthetic. Full
fine-tuning narrowly beats the three LoRA seeds (chrF 38.07, 39.44, and 38.26).
Sampling reduces repetition but also lowers chrF.

The standard `rouge_score` tokenizer is unsuitable for these Devanagari
outputs: every system scores about 0.000--0.001 with it, compared with
0.118--0.238 using the Unicode-aware tokenizer. A bounded review of six ACL
Anthology papers is recorded in `results/rouge_literature_review.md`; five did
not document a custom or language-aware ROUGE tokenizer.

## Training configuration

The reference configuration uses three epochs, seed 42, completion-only loss,
and LoRA with rank 16 and alpha 32. The study design also compares:

- context lengths 512 and 1,024;
- completion-only and full-conversation loss;
- LoRA ranks 8, 16, and 32;
- 1, 3, and 5 epochs;
- LoRA and full-parameter fine-tuning;
- three seeds for the selected configuration.

The notebook uses step-based evaluation and fixed warm-up steps. Comparisons
use the same constant-with-warm-up schedule and are rescored with a shared
1,024-token, assistant-only evaluation protocol.

| Training run | Eval loss | Token accuracy | Minutes |
|---|---:|---:|---:|
| LoRA r16, 1 epoch | 2.0534 | 0.6007 | 17.4 |
| LoRA r16, 3 epochs | 1.8945 | 0.6280 | 50.7 |
| LoRA r16, 5 epochs | 1.8472 | 0.6343 | 83.8 |
| LoRA r8, 3 epochs | 1.9591 | 0.6183 | 50.8 |
| LoRA r32, 3 epochs | 1.8578 | 0.6338 | 51.2 |
| LoRA r16, 512 tokens | 1.8175 | 0.6271 | 47.2 |
| LoRA r16, all-token loss | 1.9037 | 0.6269 | 51.0 |
| Full fine-tune, 3 epochs | **1.7021** | **0.6381** | 54.3 |

Across the three main LoRA seeds, eval loss is 1.8903 ± 0.0134 and token
accuracy is 0.6251 ± 0.0028 (sample standard deviation). Assistant-only masking
has little effect here. Rank 32 and five epochs help modestly, while full
fine-tuning gives the strongest loss and generation results.

## Artifacts

Compact training and generation results are versioned under `results/`.
Predictions and weights stay outside Git because of their size. Nothing from
this run was uploaded to Hugging Face.
