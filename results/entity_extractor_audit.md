# Person and date extractor audit

The person and date regexes in `evaluate.py` are heuristics, not a named-entity
recognizer. This audit measures the extractor itself against hand labels,
using the gold answers only (no model predictions are needed for this — the
question is whether the regex correctly reads a given piece of text, not
whether the model wrote a good one). It uses the same seeded 90/10 split
(`seed=42`) as the main evaluation and covers the first 100 of the 1,718
held-out test turns.

## Person extractor

Before this audit, `PERSON_AFTER_TITLE_RE` kept a trailing case suffix
(`राईले`) while `PERSON_CASE_RE` stripped it (`राई`), so the same person
surfaced as two distinct set members and set-intersection recall was
depressed for every system. Both patterns now strip the suffix
(`strip_person_case_suffix` in `evaluate.py`).

With that fix applied, the 100-turn sample yields 385 person-field candidates
(99 of the 100 turns produced at least one). Each candidate was read in its
sentence context and labeled correct (denotes a specific real person) or
incorrect by hand; each turn's gold text was also read independently to count
every real distinct person actually mentioned in it, whether or not the
regex caught it.

| | Count |
|---|---:|
| Candidates emitted | 385 |
| Candidates that are real people (hand-labeled) | 73 |
| **Extractor precision** | **0.19** |
| Real distinct person mentions (hand-labeled, per-turn) | 125 |
| Real mentions the extractor caught | 69 |
| **Extractor recall** | **0.55** |

Both numbers are properties of the regex, not of any model. They set a hard
ceiling on the `person` row of the entity table: even a system that
reproduced the gold answer verbatim would not score above roughly 0.19
precision / 0.55 recall on `person`, because that's as well as the extractor
reads gold text. **The `person` precision/recall/F1 in `generation_metrics.json`
should be read as bounded by, not equal to, model quality**, until the
extractor itself is replaced or this ceiling is reported alongside every
number derived from it.

### Why precision is low: the case-marker pattern is not a name marker

`PERSON_CASE_RE` matches any two Devanagari words immediately before
ले/लाई/सँग/बाट, which mark the ergative/dative/instrumental/ablative case for
*any* noun phrase, not specifically names. Most candidates are common-noun
phrases that happen to sit in that grammatical position:

- `प्रदेश सरकार` (province government), `केन्द्रीय सरकार` (central government)
- `यो निर्णय` (this decision), `यसले मतदाता` (this... voters)
- `गर्ने उद्देश्य` (the purpose of doing), `निर्वाचन क्षेत्र` (constituency)
- `देउवा पक्ष` (the Deuba faction — a real surname, but the span names a
  faction, not the person)

### Why recall is low: the marker patterns miss most real mentions

Reading the full 100-turn sample surfaced four concrete, recurring failure
modes, in addition to the precision problem above:

1. **Nominative subjects have no case marker and are invisible.**
   `गोकुलप्रसाद बाँस्कोटा काभ्रे क्षेत्र नम्बर २ बाट लगातार तीन पटक चुनाव लडेका थिए।`
   never triggers either pattern — the name is the bare subject.
2. **Comma/र-conjoined lists only mark the last item.**
   `रवि लामिछाने, बालेन्द्र शाह र कुलमान घिसिङलाई लक्षित गरेर` — only
   `कुलमान घिसिङ` is captured; the first two names carry no marker of their
   own.
3. **Single-word surnames need a title to be caught.** `PERSON_CASE_RE`
   requires a two-word span, so a bare `ओलीले` (`ओली` + `ले`) is invisible
   unless a title word from `PERSON_TITLES` immediately precedes it.
4. **को/का/की (genitive) and भन्दा (comparative) are not in the suffix
   list.** `सुवास पोखरेलका अनुसार`, `बिना गुरूङको भन्दा` — both real,
   frequent constructions in this corpus — are missed entirely.

None of these are fixed by this change; fixing them is a larger regex (or a
real NER model) rewrite, out of scope here. This audit exists so the
precision/recall numbers are reported with a known, measured ceiling instead
of taken at face value.

## Date extractor

`NEPALI_MONTH_RE` previously required the day number *before* the month name
(`२८ वैशाख`) and missed the dominant order in this corpus, month name first
(`वैशाख २८`). It now matches both orders. Re-running the (numeric `DATE_RE` +
Nepali-month) extraction over all 1,718 gold test answers:

| | Before fix | After fix |
|---|---:|---:|
| Gold dates found | 99 | 244 |

This is a mechanical fix (date formats are syntactic, unlike names), so it
was verified by rerunning the regex rather than by hand-labeling. 244 dates
over 1,718 turns is still sparse — most answers in this corpus don't state a
date — and the fix doesn't address remaining gaps such as two numbers sharing
one month (`मंसिर १ र २ गते`), where only the first is caught.

## What this does and doesn't fix

This audit is against the gold dataset only; it does not require the trained
model or its predictions. `results/generation_metrics.json` has since been
regenerated from the saved predictions with both regex fixes, so its
`person`/`date` precision, recall, and F1 values reflect the corrected
extractor. The measured extractor limitations still apply to their
interpretation.
