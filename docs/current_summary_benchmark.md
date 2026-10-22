# Current summarization benchmark — 22 October 2026

The current 4.6.0 pipeline at its default 800-word target achieved **20.94 ROUGE-L F1 points**, with a 95% case-bootstrap interval of **19.41–22.58**, on all **100 official IN-Abs test judgments**.

**This is an automatic reference-overlap score, not a percentage of legally correct summaries.** Role-classification accuracy remains a separate 83.12% development result.

## Results

All ROUGE values below are F1 × 100; parentheses contain 95% case-bootstrap intervals.

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | Mean words | Over target | Failures |
|---|---:|---:|---:|---:|---:|---:|
| pipeline_800 | 37.42 (34.69–40.18) | 18.25 (16.44–20.18) | 20.94 (19.41–22.58) | 308.3 | 1/100 | 0/100 |
| lead_800 | 48.51 (46.26–50.67) | 23.81 (21.78–25.98) | 24.13 (22.45–25.91) | 768.9 | 0/100 | 0/100 |
| pipeline_400 | 33.85 (31.15–36.58) | 16.15 (14.49–17.97) | 19.43 (17.92–21.02) | 248.6 | 9/100 | 0/100 |
| lead_400 | 40.91 (38.48–43.32) | 18.12 (16.09–20.28) | 20.21 (18.66–21.91) | 374.0 | 0/100 | 0/100 |

The current structured pipeline scores below the opening-sentence baseline on ROUGE-L. These results do not establish a reference-overlap improvement from the pipeline. The decision fixes addressed specific attribution errors; they did not establish better overall summarization.

At 800 words, the paired pipeline-minus-lead ROUGE-L difference is **-3.19 (-5.14–-1.31) points**. The two methods have the same requested budget, but different actual lengths because the app preserves complete passages and conditions.

## What the score means

ROUGE-1 compares words, ROUGE-2 compares word pairs, and ROUGE-L compares ordered word overlap. F1 balances precision and recall against the published reference summary. Each case has equal weight. No arbitrary score threshold is used to call a summary correct.

For the default pipeline, mean ROUGE-1 precision is **66.86%** and recall is **30.42%**. These percentages describe matching token counts under the scorer, not the percentage of correct facts or covered legal issues.

Exact-source checks passed for **100/100** default-pipeline outputs; **51/100** had at least one empty section. Copying source text does not prove correct attribution or complete reasoning.

The default pipeline scored **15,606 extracted sentences**, but its final briefs averaged only **308 words** despite the 800-word target. The decision section was empty in **47/100** briefs, reasoning in **17/100**, and the issue in **9/100**; these counts overlap. Empty sections are observable coverage limitations, not independently annotated legal errors. This suggests investigating selection/abstention and content retention before claiming a stronger summarizer. [Diagnostic counts](../evaluation/current_summary_diagnostics.json).

## Reliability and limits

- All 100 cases are included, including any processing failures; code and weights were frozen before generation. No pipeline tuning followed these scores.
- The exact-text audit found no overlap with all 7,030 archive training judgments. It does not clear semantic duplicates or upstream role/pretraining exposure.
- The official test set was already used in Stage 3.6. This is a reused benchmark, not a new blind sample or an unseen-document legal-accuracy measurement.
- Inputs are original publisher TXT files, so these results do not evaluate PDF extraction, OCR or the recent PDF quotation-recovery rule.
- Confidence intervals describe case variation within this benchmark, not all sources of uncertainty or performance on another domain.
- The recorded run uses the production CUDA path on Torch 2.6.0+cu124; CPU/GPU numerical equivalence has not been established.
- The previous 52.02/31.77/30.73 Stage 3.6 scores used different sentence candidates, selection and length rules. Treat those as a separate historical experiment.

## A sentence for the presentation

> “Our current structured summarizer scored 37.42 ROUGE-1, 18.25 ROUGE-2 and 20.94 ROUGE-L F1 on the 100-case IN-Abs benchmark at an 800-word target. These are automatic overlap scores; legal correctness has not been measured as an accuracy percentage.”

[Protocol and reproduction](automatic_evaluation_protocol.md) · [Machine-readable report](../evaluation/current_summary_benchmark.json) · [Frozen inputs and hashes](../evaluation/current_summary_protocol.json) · [All case scores](../evaluation/current_summary_per_case.csv)
