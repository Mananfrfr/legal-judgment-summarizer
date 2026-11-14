# Faculty report: precision, recall and F1

This report recomputes aggregates from saved evaluation records. It does not rerun the model or measure legal correctness.

## 1. Summarization: 100 reserved IN-Abs judgments

All metrics below are macro averages across the same 100 documents, shown on a 0–100 scale. ROUGE compares the summary with the dataset reference. F1 is computed per document, then averaged; it is not the harmonic mean of the two displayed averages.

| Method | Metric | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| baseline_800 | rouge1 | 65.96 | 33.86 | 41.00 |
| baseline_800 | rouge2 | 32.90 | 16.31 | 19.98 |
| baseline_800 | rougeL | 36.59 | 17.83 | 21.98 |
| lead_800 | rouge1 | 47.48 | 62.13 | 50.47 |
| lead_800 | rouge2 | 23.50 | 31.36 | 25.01 |
| lead_800 | rougeL | 23.04 | 31.43 | 24.81 |
| candidate_800 | rouge1 | 53.54 | 69.93 | 56.67 |
| candidate_800 | rouge2 | 31.72 | 41.88 | 33.62 |
| candidate_800 | rougeL | 28.30 | 38.01 | 30.23 |
| baseline_400 | rouge1 | 67.67 | 29.18 | 37.74 |
| baseline_400 | rouge2 | 33.82 | 14.18 | 18.49 |
| baseline_400 | rougeL | 38.99 | 16.10 | 21.10 |
| lead_400 | rouge1 | 55.63 | 38.35 | 42.53 |
| lead_400 | rouge2 | 24.60 | 16.80 | 18.58 |
| lead_400 | rougeL | 27.58 | 19.08 | 21.01 |
| candidate_400 | rouge1 | 65.84 | 47.66 | 51.85 |
| candidate_400 | rouge2 | 36.04 | 25.76 | 28.04 |
| candidate_400 | rougeL | 35.87 | 25.79 | 28.04 |

`baseline` = previous 4.6 pipeline; `candidate` = upgraded 4.7 pipeline; `lead` = opening-sentence baseline. Suffixes are word targets.

## 2. Sentence-role classification: recorded development results

| Model | Accuracy (%) | Macro F1 (%) | Weighted F1 (%) |
|---|---:|---:|---:|
| Stage 1 / majority | 34.18 | 3.92 | 17.41 |
| Stage 1 / knn | 25.50 | 18.40 | 21.41 |
| Stage 1 / logreg | 46.61 | 39.21 | 48.06 |
| Stage 1 / svm | 47.93 | 38.49 | 49.40 |
| Stage 1 / word_bilstm | 50.05 | 41.35 | 50.82 |
| Stage 2.2 / target256 | 68.18 | 51.66 | 67.04 |
| Stage 2.2 / context256 | 76.31 | 61.49 | 75.69 |
| Stage 2.3 / bilstm_softmax | 83.12 | 65.41 | 82.37 |
| Stage 2.3 / bilstm_crf | 83.12 | 65.16 | 82.39 |
| Stage 2.4 / bilstm_weighted_ce_seed42 | 82.70 | 64.79 | 82.11 |
| Stage 2.4 / bilstm_weighted_focal_seed42 | 81.66 | 64.81 | 81.05 |

Role precision, recall, per-class support and the confusion matrix are not supplied in this aggregate CSV. They cannot be reconstructed from accuracy/F1 alone. Recompute them from saved ground-truth labels and predictions; do not invent them.

## 3. Explain the metrics

- Precision = TP / (TP + FP): how much of what was predicted is correct.
- Recall = TP / (TP + FN): how much of the expected content was recovered.
- F1 = 2 × precision × recall / (precision + recall): balances precision and recall.
- For role classification, TP/FP/FN refer to class labels. Macro F1 gives each class equal weight; weighted F1 weights classes by their support.
- For ROUGE-1/2, precision and recall count overlapping unigrams/bigrams. ROUGE-L uses longest-common-subsequence overlap. These are textual overlap metrics, not factual-correctness rates.

## 4. Evidence and limitations

- Candidate-800 ROUGE-L F1: 30.23 (95% case-bootstrap interval 28.57–31.96). Previous pipeline: 21.98; lead: 24.81, on these same cases.
- 100/100 candidate summaries passed exact-source checks; 13/100 had an empty decision section. Neither 100% source matching nor 87% decision availability means summary accuracy.
- The sample excludes known selector train/dev IDs and exact normalized duplicates. It remains same-corpus; upstream training overlap and semantic near duplicates are not cleared.
- The initial freeze was amended for a synthetic quotation test before reserved scores were inspected. Both freezes and the amendment are preserved in evaluation/.
- Existing weights were unchanged. The earlier official-test ROUGE-L 20.94 is from a different sample and must not be used as the before value for this table.
- Outcome/amount/deadline rule checks are diagnostic screens, not a new human-validated factuality metric. No LLM judge score is included.

## 5. What to say to faculty

> Our role classifier recorded 65.41% macro F1 and 82.37% weighted F1 on development data. Our upgraded summarizer recorded 56.67 ROUGE-1 F1, 33.62 ROUGE-2 F1 and 30.23 ROUGE-L F1 on 100 reserved cases. We separately check source evidence and screen decisions, amounts and deadlines for possible errors. These checks do not establish a legal-correctness percentage.

## Reproduction

Run `python scripts/build_faculty_metrics_report.py` from the repository. No model download, API key or third-party package is required.

Inputs: `evaluation/coverage_upgrade_per_case.csv`, `evaluation/coverage_upgrade_reserved.json`, `evaluation/coverage_upgrade_protocol.json`, `evaluation/role_results.csv`.
