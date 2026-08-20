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
  entity-overlap metrics, with a conversation-level cluster bootstrap
  (turns within a conversation are correlated, so resampling turns
  independently understates CI width) and a paired cluster-bootstrap test
  for comparing two systems directly (`evaluate.py --compare A B`).

The regex-based person/date entity extractor is a heuristic, not a NER
model, and it is audited against hand labels in
`results/entity_extractor_audit.md`: on a 100-turn hand-labeled sample it
gets 0.19 precision / 0.55 recall against real person mentions in gold
text, which puts a ceiling on how the `person` row below can be read. The
date pattern previously required the day number before the month name and
missed the dominant month-first order in this corpus; both orders are now
matched.

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
category-level entity counts are in `results/generation_metrics.json`. The
file includes the cluster bootstrap, entity-suffix and date-regex fixes,
gold-answer language-ID reference, entity F1, and paired comparison described
below.

| System | chrF | Unicode ROUGE-L | Nepali | Entity P/R/F1 | rep-4 |
|---|---:|---:|---:|---:|---:|
| Base | 17.09 | 0.131 | 72.4% | 0.323 / 0.095 / 0.146 | 0.065 |
| Nepali instruction | 15.96 | 0.126 | 73.1% | 0.317 / 0.089 / 0.138 | 0.054 |
| Three-shot Nepali | 16.69 | 0.126 | 82.1% | 0.257 / 0.087 / 0.130 | 0.068 |
| Character TF-IDF retrieval | 32.73 | 0.158 | 100.0% | 0.126 / 0.126 / 0.126 | 0.000 |
| Copy question | 8.50 | 0.118 | 69.9% | 0.524 / 0.062 / 0.111 | 0.000 |
| LoRA, seed 42 | 38.07 | 0.229 | 99.6% | 0.235 / 0.215 / 0.225 | 0.037 |
| Full fine-tune | 39.41 | 0.238 | 100.0% | 0.248 / 0.226 / 0.236 | 0.029 |
| LoRA, sampled | 37.20 | 0.215 | 99.5% | 0.208 / 0.206 / 0.207 | **0.011** |

Full fine-tune is not bolded as the winner: its chrF CI is [38.80, 40.03] and
LoRA seed 7's is [38.85, 40.02]. The paired full-minus-LoRA difference is
-0.028 chrF with a 95% CI of [-0.337, 0.272], which includes zero. The runs
are not distinguishable on this test set.

**Copy-question's entity precision (0.524, the highest in the table) is a
volume artifact, not a factual-accuracy result.** It echoes the question
back, so almost everything it emits is a real entity — it just emits very
few compared to gold (785 person predictions versus 6,222 gold mentions;
the base model predicts 1,795, for scale). The base model's 0.323 similarly beats the
fine-tuned model's 0.248 partly because the base model says less. F1 is the
column to read for a precision/recall tradeoff comparison; it still does not
fully remove the volume effect, which is why the number above is reported
alongside it rather than in place of it.

**The character TF-IDF retrieval baseline (32.73 chrF, zero training) is the
most important line in this table and was previously under-discussed.**
Against it, fine-tuning's gain is 39.41 − 32.73 = 6.7 chrF, not the 22-point
gain visible against the base model (39.41 − 17.09). Both deltas are worth
reporting: fine-tuning clearly beats a zero-cost lookup table, but by much
less than it beats doing nothing. A nearest-neighbour lookup over training
questions scoring 32.7 chrF on held-out test questions also means the random
90/10 split has enough near-duplicate question content that a lookup table
gets most of the way there — evidence the split is not source-disjoint,
not evidence about the model. Fine-tuning's clearest advantage over the
retrieval baseline is entity recall (0.215 vs. 0.127), which is a lower-cost
claim than "produces a large language-control gain" and is the one this
report leans on.

Assistant-only masking made little difference to loss (see below); sampling
reduces repetition but also lowers chrF.

The standard `rouge_score` tokenizer is unsuitable for these Devanagari
outputs: every system scores about 0.000--0.001 with it, compared with
0.118--0.238 using the Unicode-aware tokenizer. A bounded review of six ACL
Anthology papers is recorded in `results/rouge_literature_review.md`; five did
not document a custom or language-aware ROUGE tokenizer.

`copy-question` echoes the Nepali question verbatim and should score close to
100% on the Nepali-language-ID check; it scores 69.9%, well below every
fine-tuned system. That gap says fastText `lid.176` is not reliable on short
Devanagari text (it is known to confuse `ne`/`hi`), so the 99.6–100.0% rows
for the fine-tuned systems should be read as "this checker did not flag a
problem," not as a validated 100% Nepali rate. `evaluate.py` now also scores
the gold answers themselves when a LID model is supplied
(`_gold_language_reference` in the report) so the ceiling is visible next to
every system's number. Gold answers score 99.88% Nepali and 0.12% Hindi.

### Corrected re-evaluation

The published metrics were regenerated from the saved full-test predictions
after adding the cluster bootstrap, paired comparison, person-suffix and date
fixes, gold-language-ID reference, and entity F1. The raw predictions and
weights remain outside Git (see `.gitignore`: `runs/`,
`evaluation/*.jsonl`).

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
