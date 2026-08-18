# ROUGE tokenization in Indic-language papers

This is a small audit, not a systematic review. Six peer-reviewed ACL Anthology
papers from 2020–2025 were selected because they evaluate generated text in an
Indic language with ROUGE. The audit records whether the paper identifies a
language-aware or custom ROUGE tokenizer. It does not infer undocumented code.

| Paper | Language | Tokenization reported |
|---|---|---|
| [Dhakal and Baral (2025)](https://aclanthology.org/2025.chipsal-1.12/) | Nepali | None |
| [T K et al. (2024)](https://aclanthology.org/2024.icon-1.11/) | Tamil | None |
| [Lal et al. (2023)](https://aclanthology.org/2023.icon-1.58/) | Hindi | None |
| [Khan et al. (2023)](https://aclanthology.org/2023.banglalp-1.10/) | Bangla | None |
| [Roychowdhury et al. (2022)](https://aclanthology.org/2022.icon-main.40/) | Bangla | ROUGE 1.5.5; no custom tokenizer described |
| [Kumar et al. (2022)](https://aclanthology.org/2022.emnlp-main.360/) | 11 Indic languages | Multilingual ROUGE with language-specific segmentation |

Five of the six papers do not report language-aware or custom tokenization.
This is a documentation finding: it does not establish that those authors used
an ASCII-only tokenizer. The experiment in this repository demonstrates the
narrower implementation problem directly by scoring identical Nepali outputs
with `rouge_score`'s default tokenizer and a Unicode-aware tokenizer.
