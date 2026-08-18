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
training-time truncation in this dataset. Whether that additional context
improves evaluation loss remains to be established by the 512-versus-1,024
ablation.

## Evaluation protocol

The primary comparison uses every held-out conversation. Generated turns are
evaluated against the reference answer while both generative models receive the
same reference history. Greedy decoding is used for the reproducible headline
result; sampled decoding is reported separately when testing repetition.

The planned main table contains:

1. base Gemma;
2. base Gemma with a Nepali news-style system instruction;
3. base Gemma with three Nepali demonstrations;
4. character n-gram TF-IDF nearest-neighbour retrieval;
5. question copying;
6. the fine-tuned model.

Results are not copied from stale notebook output. Tables should be added only
after all systems have been evaluated under the same protocol, with bootstrap
confidence intervals over test turns.

The two deterministic baselines have been run on all 1,718 held-out turns. The
copy baseline obtains chrF 8.50 and the character TF-IDF nearest-neighbour
baseline obtains chrF 32.73. Default-tokenizer ROUGE-L is 0.0004 and 0.0000,
respectively, while Unicode-aware ROUGE-L is 0.1185 and 0.1579. These results
are stored in `results/text_baselines.json`; generative systems remain pending.

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
across epoch counts must account for the learning-rate schedule.

## Current status

The corrected token statistics and deterministic baseline results are recorded
under `results/`. The complete generative baseline table, training ablations,
and adapter release remain pending fresh execution. Earlier qualitative output
suggests that fine-tuning changes the output language and register, but this is
not presented as a factuality result.
