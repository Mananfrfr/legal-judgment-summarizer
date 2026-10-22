# Decision extraction upgrade — 22 October 2026

Version **4.6.0** improves the review-assisted pipeline. It does not retrain the neural model. Recorded sentence-role DEV accuracy remains **83.12%**, macro F1 **65.41%**; current end-to-end legal accuracy is **not independently measured**.

## What changed and why

1. **Read the real final decision.** Optional commas in “The appeal is, accordingly, allowed” and batch-of-matters wording no longer hide the disposition. Only temporary matching text is normalized; displayed evidence remains verbatim.
2. **Separate earlier rulings and proposals.** Passive earlier-court reports cannot independently anchor the current decision. “Ought to be dismissed” is a proposal, not a completed disposition. Explicit current reversals remain eligible.
3. **Keep important qualifications.** Adjacent court statements that no view is expressed on the merits stay with the final decision. Attribution checks prevent a party's claim from becoming a court qualification. Existing conditions, amounts and deadlines remain grouped.
4. **Recover a swallowed final PDF section.** One 111-page judgment had an unclosed inline dictionary quotation. Recovery requires native PDF geometry, an established Roman outline, matching font/margin and a later numbered final heading. It does not reset ordinary block quotations, arbitrary headings or plain-text outlines. The output explicitly requests attribution review; this heuristic cannot prove authorship.
5. **Make the app easier to launch and verify.** `Start-App.ps1` checks model paths and starts localhost with the available isolated environment. Health reports the actual pipeline version.

The affected inference modules are `stage44_source.py`, `stage44_attribution.py` and `stage44_brief.py`. Encoder, selector, learned features, weights and model-window scoring are unchanged. The four presentation notebooks remain concise; Notebook 04 points to these changes.

## Measured verification

| Check | Before | Revised | Interpretation |
|---|---:|---:|---|
| Automated suite | 71 at consolidation | 86 pass | Software regression tests, including adversarial attribution/quote cases |
| Three previously inspected judgments | 1/3 | 3/3 | Specified decision checks only |
| Earlier ten development judgments | 29/29 | 29/29 | Specified content checks retained |
| Earlier reasoning/review checks | 15/15 | 15/15 | Existing review regressions retained |
| Four additional judgments | 2/4 | 4/4 after tuning | Specified operative-text coverage, not overall correctness or unseen accuracy |

The four additional official-source PDFs total **228 pages and 1,679 sentences**. All sentences were scored with the actual trained model. Baseline, frozen candidate `bbcb718` and revised rules use identical predictions, with source/sentence hashes verified. No new weights were trained. The source targets were recorded after reading the final source pages and before inspecting model outputs. The purposive sample was not random, independent or proven absent from training; search snippets exposed some outcomes. First-run baseline and candidate each passed 2/4. Those failures motivated the revision, so all four are now **development cases**.

The first run missed the Cuddalore no-merits qualification and selected earlier reported proceedings in APTEL. The revision retains the qualification and selects APTEL's page-111 final order and other-remedies qualification. Source matching passes for all selected evidence in these checks; source matching alone does not establish legal correctness.

Reports: [three cases](../evaluation/decision_upgrade_development.json), [earlier ten](../evaluation/decision_upgrade_previous_documents.json), [review checks](../evaluation/decision_upgrade_previous_review.json), [four-case before/after report with source URLs and hashes](../evaluation/additional_validation_results.json), [selection plan](../evaluation/additional_validation_plan.json), [source targets](../evaluation/additional_source_targets.json).

## Remaining errors and limits

- The APTEL **issue section still selects a question from a cited case**. Correcting its decision does not correct the whole brief.
- The earlier Section 19 reasoning omission remains outside this decision-focused fix.
- OCR, quotations, attribution and long-sentence truncation can still cause errors. The model ranks the first 192 target tokens for long sentences; evidence retains the complete source text.
- Independent human completeness/attribution annotation and a new untouched evaluation set remain necessary for a general accuracy claim. No-warning output also needs review.

This release is suitable for a faculty demonstration and assisted review. It is not an independently validated autonomous legal-summary service. The original failed outputs/review remain historical evidence, not erased results.

## Reproduce

```bash
python -m unittest discover -s tests -v
python scripts/evaluate_decision_upgrade.py --pdf-dir PATH_TO_ORIGINAL_PDFS --cache-dir PATH_TO_FROZEN_RESULTS --output outputs/new_decision_replay
```

For the four public-source cases, use `scripts/run_additional_validation.py`: `--mode prepare` downloads and freezes source/module hashes; `--mode score --encoder models/encoder --bundle models/stage4_english_model_bundle.zip` scores them; `--mode replay --label NAME --source-dir PATH_TO_SRC` compares extraction rules with those cached scores. Supply `--plan evaluation/additional_validation_plan.json --output outputs/additional_validation` for each command. Use separate output folders for new experiments. To reproduce the historical first run, check out `bbcb718` source before preparation/scoring; baseline source is `10099ab`. Keep the runner from 4.6 and pass `--source-dir` explicitly. Some environments may need to download the official APTEL PDF through their browser because of issuer-chain configuration; never disable TLS verification.

The comparison report expects labels `baseline`, `candidate_initial`, and `revised_final`:

```bash
python scripts/report_additional_validation.py --runs outputs/additional_validation --output evaluation/additional_validation_results.json
```

Regeneration requires the original frozen run files or the corresponding source versions, not merely running current rules three times. Older supplied PDFs/predictions remain local; the public report provides their checksums without publishing private uploads.
