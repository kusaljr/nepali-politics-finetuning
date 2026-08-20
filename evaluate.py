"""Evaluate saved turn-level predictions and deterministic text baselines.

Prediction files are JSONL records with: system, question, prediction, and gold.
Use --build-text-baselines to create copy and character TF-IDF retrieval outputs
from the same seeded split as the notebook.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import regex
import sacrebleu
from datasets import Dataset
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer


class NepaliTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return regex.findall(r"[\p{L}\p{M}\p{N}]+", text.lower())


UNICODE_ROUGE = rouge_scorer.RougeScorer(
    ["rougeL"], tokenizer=NepaliTokenizer()
)
DEFAULT_ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
TOKEN_RE = regex.compile(r"[\p{L}\p{M}\p{N}]+")
NUMBER_RE = regex.compile(r"\p{N}+(?:[.,]\p{N}+)*")
DATE_RE = regex.compile(
    r"(?:\p{N}{1,4}[-/.]\p{N}{1,2}(?:[-/.]\p{N}{1,4})?)"
)
NEPALI_MONTHS = (
    "बैशाख", "वैशाख", "जेठ", "असार", "साउन", "भदौ", "असोज",
    "कात्तिक", "मंसिर", "पुस", "माघ", "फागुन", "चैत",
)
# Both attested orders: "२८ वैशाख" (number first) and "वैशाख २८" (month
# first, the dominant order in this corpus's news-style dates).
NEPALI_MONTH_RE = regex.compile(
    rf"\p{{N}}{{1,2}}\s+(?:{'|'.join(NEPALI_MONTHS)})"
    rf"|(?:{'|'.join(NEPALI_MONTHS)})\s+\p{{N}}{{1,2}}"
)
PARTY_TERMS = (
    "कांग्रेस", "एमाले", "माओवादी", "रास्वपा", "राप्रपा", "जसपा",
    "लोसपा", "एकीकृत समाजवादी", "जनमत", "नागरिक उन्मुक्ति",
)
PERSON_TITLES = (
    "राष्ट्रपति", "पूर्वराष्ट्रपति", "प्रधानमन्त्री", "उपप्रधानमन्त्री",
    "मन्त्री", "सांसद", "अध्यक्ष", "महासचिव", "नेता", "उम्मेदवार",
    "मेयर", "प्रमुख", "प्रवक्ता",
)
DEVANAGARI_WORD = r"[\p{Devanagari}][\p{Devanagari}\p{M}]*"
PERSON_AFTER_TITLE_RE = regex.compile(
    rf"(?:{'|'.join(PERSON_TITLES)})\s+({DEVANAGARI_WORD}(?:\s+{DEVANAGARI_WORD})?)"
)
PERSON_CASE_RE = regex.compile(
    rf"({DEVANAGARI_WORD}\s+{DEVANAGARI_WORD})(?:ले|लाई|सँग|बाट)\b"
)
# PERSON_AFTER_TITLE_RE's word class swallows a trailing case marker
# (e.g. "राईले") because ले/लाई/सँग/बाट are themselves Devanagari
# characters; PERSON_CASE_RE excludes the marker by construction. Strip
# it from both so the same person doesn't surface as two distinct
# entities depending on which pattern caught them.
PERSON_CASE_SUFFIX_RE = regex.compile(r"(?:ले|लाई|सँग|बाट)$")


def strip_person_case_suffix(name: str) -> str:
    return PERSON_CASE_SUFFIX_RE.sub("", name)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def conversation_turns(conversations: list[list[dict]]) -> list[dict]:
    rows = []
    for conversation_id, messages in enumerate(conversations):
        pending = None
        for message in messages:
            if message["role"] == "user":
                pending = message["content"]
            elif pending is not None:
                rows.append({
                    "conversation_id": conversation_id,
                    "question": pending,
                    "gold": message["content"],
                })
                pending = None
    return rows


def build_text_baselines(dataset_path: Path, output_path: Path, seed: int) -> None:
    conversations = read_jsonl(dataset_path)
    split = Dataset.from_list(
        [{"messages": conversation} for conversation in conversations]
    ).train_test_split(test_size=0.1, seed=seed)
    train = conversation_turns([row["messages"] for row in split["train"]])
    test = conversation_turns([row["messages"] for row in split["test"]])

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2)
    train_matrix = vectorizer.fit_transform(row["question"] for row in train)
    test_matrix = vectorizer.transform(row["question"] for row in test)
    similarities = test_matrix @ train_matrix.T
    nearest = np.asarray(similarities.argmax(axis=1)).ravel()

    output = []
    for row, neighbour in zip(test, nearest):
        common = {
            "conversation_id": row["conversation_id"],
            "question": row["question"],
            "gold": row["gold"],
        }
        output.append({**common, "system": "copy-question", "prediction": row["question"]})
        output.append({
            **common,
            "system": "char-tfidf-nearest-neighbour",
            "prediction": train[int(neighbour)]["gold"],
        })
    write_jsonl(output_path, output)


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def distinct_2(text: str) -> float:
    bigrams = ngrams(TOKEN_RE.findall(text.lower()), 2)
    return len(set(bigrams)) / len(bigrams) if bigrams else 0.0


def rep_4(text: str) -> float:
    fourgrams = ngrams(TOKEN_RE.findall(text.lower()), 4)
    return 1.0 - len(set(fourgrams)) / len(fourgrams) if fourgrams else 0.0


def entities(text: str) -> dict[str, set[str]]:
    dates = set(DATE_RE.findall(text)) | set(NEPALI_MONTH_RE.findall(text))
    numbers = set(NUMBER_RE.findall(text)) - {
        part for date in dates for part in NUMBER_RE.findall(date)
    }
    persons = {strip_person_case_suffix(name) for name in PERSON_AFTER_TITLE_RE.findall(text)}
    persons.update(strip_person_case_suffix(name) for name in PERSON_CASE_RE.findall(text))
    return {
        "person": persons,
        "party": {term for term in PARTY_TERMS if term in text},
        "date": dates,
        "number": numbers,
    }


def entity_counts(prediction: str, gold: str) -> dict[str, tuple[int, int, int]]:
    pred_entities, gold_entities = entities(prediction), entities(gold)
    return {
        kind: (
            len(pred_entities[kind] & gold_entities[kind]),
            len(pred_entities[kind]),
            len(gold_entities[kind]),
        )
        for kind in pred_entities
    }


def _cluster_groups(values: list[float], conversation_ids: list[int]) -> list[np.ndarray]:
    by_conversation: dict[int, list[float]] = defaultdict(list)
    for value, conversation_id in zip(values, conversation_ids):
        by_conversation[conversation_id].append(value)
    return [np.asarray(group, dtype=float) for group in by_conversation.values()]


def cluster_bootstrap_mean(
    values: list[float], conversation_ids: list[int], seed: int, samples: int = 10_000,
) -> tuple[float, float]:
    """Resample conversations (not turns) with replacement.

    Turns within a conversation share topic and phrasing, so resampling
    turns independently understates variance and produces CIs that are
    too narrow. Resampling whole conversations respects that clustering.
    """
    groups = _cluster_groups(values, conversation_ids)
    group_sums = np.asarray([group.sum() for group in groups])
    group_sizes = np.asarray([len(group) for group in groups])
    rng = np.random.default_rng(seed)
    choice = rng.integers(0, len(groups), size=(samples, len(groups)))
    means = group_sums[choice].sum(axis=1) / group_sizes[choice].sum(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def paired_cluster_bootstrap_diff(
    values_a: list[float],
    values_b: list[float],
    conversation_ids: list[int],
    seed: int,
    samples: int = 10_000,
) -> dict:
    """CI on the per-turn difference (a - b), resampled by conversation.

    Two systems can each have a plausible-looking CI on their own mean
    while still overlapping heavily; this tests the paired difference
    directly instead of eyeballing overlap between separate intervals.
    """
    diffs = [a - b for a, b in zip(values_a, values_b)]
    groups = _cluster_groups(diffs, conversation_ids)
    group_sums = np.asarray([group.sum() for group in groups])
    group_sizes = np.asarray([len(group) for group in groups])
    rng = np.random.default_rng(seed)
    choice = rng.integers(0, len(groups), size=(samples, len(groups)))
    means = group_sums[choice].sum(axis=1) / group_sizes[choice].sum(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "mean_diff": float(np.mean(diffs)),
        "ci95": [float(low), float(high)],
        "excludes_zero": bool(low > 0 or high < 0),
    }


def f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def language_id_rates(texts: list[str], lid_model) -> dict:
    predicted_labels, _ = lid_model.predict([text.replace("\n", " ") for text in texts], k=1)
    labels = [item[0] for item in predicted_labels]
    return {
        "ne_accuracy": sum(label == "__label__ne" for label in labels) / len(labels),
        "hi_rate": sum(label == "__label__hi" for label in labels) / len(labels),
    }


def evaluate(rows: list[dict], seed: int, lid_model=None) -> dict:
    metrics = defaultdict(list)
    conversation_ids = [row["conversation_id"] for row in rows]
    entity_totals = defaultdict(lambda: [0, 0, 0])
    for row in rows:
        prediction, gold = row["prediction"], row["gold"]
        metrics["rougeL_default"].append(
            DEFAULT_ROUGE.score(gold, prediction)["rougeL"].fmeasure
        )
        metrics["rougeL_unicode"].append(
            UNICODE_ROUGE.score(gold, prediction)["rougeL"].fmeasure
        )
        metrics["chrf"].append(sacrebleu.sentence_chrf(prediction, [gold]).score)
        metrics["distinct_2"].append(distinct_2(prediction))
        metrics["rep_4"].append(rep_4(prediction))
        for kind, counts in entity_counts(prediction, gold).items():
            for index, count in enumerate(counts):
                entity_totals[kind][index] += count

    result = {"n": len(rows)}
    for name, values in metrics.items():
        low, high = cluster_bootstrap_mean(values, conversation_ids, seed)
        result[name] = {"mean": float(np.mean(values)), "ci95": [low, high]}
    overall = [sum(counts[i] for counts in entity_totals.values()) for i in range(3)]
    overall_precision = overall[0] / overall[1] if overall[1] else 0.0
    overall_recall = overall[0] / overall[2] if overall[2] else 0.0
    result["entity_precision"] = overall_precision
    result["entity_recall"] = overall_recall
    result["entity_f1"] = f1(overall_precision, overall_recall)
    result["entities"] = {}
    for kind, (tp, pred_count, gold_count) in entity_totals.items():
        precision = tp / pred_count if pred_count else 0.0
        recall = tp / gold_count if gold_count else 0.0
        result["entities"][kind] = {
            "precision": precision,
            "recall": recall,
            "f1": f1(precision, recall),
            "predicted": pred_count,
            "gold": gold_count,
        }
    if all("hit_max_new_tokens" in row for row in rows):
        result["generation_cap_rate"] = sum(
            row["hit_max_new_tokens"] for row in rows
        ) / len(rows)
    if lid_model is not None:
        rates = language_id_rates([row["prediction"] for row in rows], lid_model)
        result["language_ne_accuracy"] = rates["ne_accuracy"]
        result["language_hi_rate"] = rates["hi_rate"]
    return result


def paired_chrf_comparison(rows_a: list[dict], rows_b: list[dict], seed: int) -> dict:
    by_key_b = {(row["conversation_id"], row["question"]): row for row in rows_b}
    aligned_a, aligned_b = [], []
    for row in rows_a:
        match = by_key_b.get((row["conversation_id"], row["question"]))
        if match is not None:
            aligned_a.append(row)
            aligned_b.append(match)
    conversation_ids = [row["conversation_id"] for row in aligned_a]
    chrf_a = [sacrebleu.sentence_chrf(row["prediction"], [row["gold"]]).score for row in aligned_a]
    chrf_b = [sacrebleu.sentence_chrf(row["prediction"], [row["gold"]]).score for row in aligned_b]
    result = paired_cluster_bootstrap_diff(chrf_a, chrf_b, conversation_ids, seed)
    result["n"] = len(aligned_a)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path, nargs="?")
    parser.add_argument("--build-text-baselines", action="store_true")
    parser.add_argument("--dataset", type=Path, default=Path("nepali_politics_news.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("text_baselines.jsonl"))
    parser.add_argument(
        "--lid-model", type=Path,
        help="path to fastText lid.176.bin for Nepali/Hindi output rates",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--compare", action="append", nargs=2, metavar=("SYSTEM_A", "SYSTEM_B"),
        help="paired cluster-bootstrap chrF comparison between two systems; repeatable",
    )
    args = parser.parse_args()

    if args.build_text_baselines:
        build_text_baselines(args.dataset, args.output, args.seed)
        print(f"wrote {args.output}")
    if args.predictions:
        lid_model = None
        if args.lid_model:
            import fasttext
            lid_model = fasttext.load_model(str(args.lid_model))
        grouped = defaultdict(list)
        for row in read_jsonl(args.predictions):
            grouped[row["system"]].append(row)
        report = {
            name: evaluate(rows, args.seed, lid_model)
            for name, rows in grouped.items()
        }
        if lid_model is not None:
            any_rows = next(iter(grouped.values()))
            gold_by_key = {(row["conversation_id"], row["question"]): row["gold"] for row in any_rows}
            report["_gold_language_reference"] = language_id_rates(list(gold_by_key.values()), lid_model)
        if args.compare:
            report["_comparisons"] = {
                f"{system_a}_vs_{system_b}": paired_chrf_comparison(
                    grouped[system_a], grouped[system_b], args.seed
                )
                for system_a, system_b in args.compare
            }
        print(json.dumps(report, ensure_ascii=False, indent=2))
    if not args.build_text_baselines and not args.predictions:
        parser.error("provide a prediction file or --build-text-baselines")


if __name__ == "__main__":
    main()
