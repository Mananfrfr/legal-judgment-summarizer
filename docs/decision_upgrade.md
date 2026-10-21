# Decision extraction candidate — 22 October 2026

This branch is **4.6.0-preview**, a targeted pipeline revision. The released app and main branch remain Stage 4.5. Model weights were not retrained; the recorded role-classification accuracy remains **83.12% on DEV**, with macro F1 **0.6541**. There is no independently measured end-to-end summarizer accuracy.

## What was wrong

1. The disposition matcher did not recognize optional commas in `The appeal is, accordingly, allowed.` It also missed a `present batch of matters` subject.
2. Passive historical wording such as `dismissed by a Division Bench of the High Court` could become the current-decision anchor. This could select early procedural history while omitting the actual final relief.

## Candidate changes

- Normalize commas only in temporary cue-matching text. Stored sentences, character offsets, amounts and source evidence remain verbatim.
- Recognize the narrow batch/group-of-matters disposition form.
- Identify passive earlier-court dispositions in attribution and reject them as independent current-order anchors. An explicit current reversal remains eligible.
- Retain existing paragraph-level conditions, qualifications and source checks.

The changes concern `stage44_attribution.py` and `stage44_brief.py`. No encoder, selector, PDF parser, learned feature or training hyperparameter changed.

## Verification

- **76 automated tests passed**, including five new test methods covering comma-separated outcomes, passive reports, conditional relief, batch orders, negation, submissions, questions and explicit reversals.
- On the three previously inspected judgments, the specified decision checks passed for **1/3 baseline outputs and 3/3 candidate outputs**. These counts describe only the specified checks, not correctness of the entire summaries.
- The pension case now includes dismissal, Medical Board/Tribunal qualifications, the fifteen-year condition and no-costs disposition. The detention case now includes the current setting-aside order, conditional release, appeal allowance and costs deadline instead of the earlier High Court dismissal.
- All selected source spans still match the extracted text. The original ten development documents retain **29/29 targeted checks**, plus **15/15 review checks**.
- Cached model predictions were reused only after validating source, sentence, model-module and encoder hashes. This is a controlled comparison of extraction rules using the same predictions, not a new neural-model run.

See [three-case report](../evaluation/decision_upgrade_development.json), [earlier document checks](../evaluation/decision_upgrade_previous_documents.json), and [earlier review checks](../evaluation/decision_upgrade_previous_review.json).

## What this does not establish

All these documents have now been inspected and are development data. The original three-case review remains unchanged as historical evidence. Fixing these examples does not establish generalization or a new accuracy percentage. Remaining problems include issue selection, the Section 19 reasoning omission in the third case, quotation/attribution ambiguity, OCR and uncalibrated confidence. Even a brief with no automatic warning still requires human review.

Before promoting the candidate, obtain independent attribution/completeness annotations and assess a new untouched document set. Measure current-decision correctness, retained conditions, unsupported attribution and omissions alongside ROUGE. Improvements to role macro F1 should be evaluated separately; CRF and weighted-loss experiments have already been tried and did not beat the original macro F1.

## Reproduce the three-case development replay

Use the original local PDFs and frozen Stage 4.5 prediction/summary files. The replay fails if their hashes do not match; it does not silently use different weights.

```bash
python scripts/evaluate_decision_upgrade.py --pdf-dir PATH_TO_PDFS --cache-dir PATH_TO_FROZEN_RESULTS --output outputs/new_decision_replay
python -m unittest discover -s tests -v
```

This preview is kept separate from the faculty release while broader evaluation remains outstanding.
