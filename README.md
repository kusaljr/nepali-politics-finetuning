# Fine-tuning Gemma on Nepali Election News Q&A

this is my learning project where i fine-tune a small language model to answer questions about nepali politics and elections. i am just exploring how supervised fine-tuning works.

## what i did

i took `google/gemma-3-270m-it` (a small 270M parameter model) and fine-tuned it on nepali election question answer conversations. the base model was answering in hindi even when asked in nepali, so wanted to fix that.

## dataset

i made a synthetic dataset `nepali_politics_news.jsonl` of multi-turn conversations about nepali politics and elections. each conversation has around 5 turns of questions and answers in nepali (devanagari script).

- total conversations: 3460
- train split: 3114
- test split: 346

## main things i learned

**completion only loss** - by default the model trains on the full conversation including user questions which is wrong. you only want it to learn how to write answers not questions. fixed this by patching the chat template with `{% generation %}` tags.

**token length matters** - my conversations are in devanagari script which uses several tokens per word. the old code used max_length=512 which was cutting most conversations in the middle. measured the actual lengths first, most conversations needed around 800-1000 tokens, so set max_length=1024.

**LoRA instead of full fine-tuning** - trained small adapter matrices instead of all 268M weights. uses much less memory and trains faster. used r=16, lora_alpha=32.

**multi turn evaluation** - when evaluating, feed the gold history for each turn so later questions have the context they need. the old way was asking each question without any context which doesnt work for followup questions.

## results

base model was answering in hindi mixed with some bengali for nepali questions. after fine-tuning it answers in proper nepali.

example:

```
Q: सुदूरपश्चिममा लगानी सम्मेलन कहिले हुँदैछ?

BASE: सुदूरपश्चिममा लगानी सम्मेलन कहिले जाने की संभावना है। (answering in hindi)

FINE-TUNED: सुदूरपश्चिम प्रदेशका विभिन्न जिल्लामा लगानी सम्मेलन आगामी फागुन ... (proper nepali)
```

also measured ROUGE-L and chrF scores. fine-tuned model clearly beats base on both metrics.

**note:** the model learned the style and format of answers, not actual facts. so it writes fluent nepali political news style answers but sometimes makes up specific details like vote counts or names. thats expected for SFT. a 270M model is just too small to memorize thousands of specific facts like vote counts and names. if we want better factual answers we would need a much bigger model, or train on way more data that repeats the same facts many times so the model actually remembers them.

## training config

```
model: google/gemma-3-270m-it
lora r: 16
lora alpha: 32
epochs: 3
batch size: 8 (effective 16 with grad accumulation)
learning rate: 2e-4
max length: 1024
```

training took around 322 seconds on a NVIDIA RTX PRO 6000.
