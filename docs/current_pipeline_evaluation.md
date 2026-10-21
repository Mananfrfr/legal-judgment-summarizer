# Separate-document evaluation — 20 October 2026

The frozen Stage 4.5 pipeline processed all three documents and matched every selected evidence span to extracted source text. **Two of the three briefs had serious decision-section failures.** Exact copying is not proof of correct attribution or completeness.

This is a small development-excluded check. The documents were selected before outputs were inspected, and the core module hashes still match the pre-run freeze. Training-set overlap is unknown. The qualitative review below was performed by the assistant, not an independent legal reviewer; no general accuracy percentage is claimed. All documents were searchable PDFs; scan/OCR robustness was not evaluated on this set.

| Document | Pages | Sentences scored | Exact evidence | Decision finding |
|---|---:|---:|---|---|
| 129_2025_4_1501_74249_Judgement_15-Sep-2026.pdf | 93 | 565/565 | Pass | Critical failure |
| 37437_2026_6_1501_74756_Judgement_29-Sep-2026.pdf | 41 | 448/448 | Pass | Critical failure |
| 38471_2025_14_1502_75449_Judgement_16-Oct-2026.pdf | 17 | 91/91 | Pass | Final dismissal captured; other omissions remain |

## 129_2025_4_1501_74249_Judgement_15-Sep-2026.pdf

Decision section abstains despite dismissal and operative qualifications on page 93. Much of paragraph 39 appears under reasoning. No-costs disposition is absent. Explicit missing-decision warning is correct; the brief is still incomplete.

Compared source pages: 92, 93. Automatic flags: MISSING_SECTION, POSSIBLE_OMITTED_DIRECTIONS.

## 37437_2026_6_1501_74756_Judgement_29-Sep-2026.pdf

Decision section contains the earlier High Court dismissal and appellant submissions from pages 5–6. Current relief at paragraph 67/page 40 is omitted: orders set aside, release subject to custody in another case, appeal allowed. Costs of Rs.10 lakh within one month appear only among omission candidates. The issue extract is a vague reference rather than a useful question. The automatic flags do not specifically detect the wrong-court decision.

Compared source pages: 5, 6, 40, 41. Automatic flags: POSSIBLE_OMITTED_DIRECTIONS.

## 38471_2025_14_1502_75449_Judgement_16-Oct-2026.pdf

Dismissal and disposal of pending applications match the final page. Issue section is only procedural context and is flagged. Reasoning covers Section 17A but omits the separate Section 19 sanction analysis on page 16. This is not a complete or independently validated brief.

Compared source pages: 16, 17. Automatic flags: POSSIBLE_OMITTED_DIRECTIONS, ISSUE_CONTEXT_ONLY, ENCODER_TRUNCATION_REVIEW.

## Reproduce and review

Original PDF URLs are in `source_urls.json`; hashes and the pre-run criteria are in `plan_and_freeze.json`. The release includes `results/heldout_batch.zip` with machine extracts, evidence JSON and audits. Use `independent_review_template.csv` for separate human ratings; it is deliberately blank. Re-run inference on the original PDFs using the same model fingerprint and core hashes before comparing any new model.

## Release decision

Release only as a local, human-reviewed research app. Do not claim unseen-document accuracy or publish an autonomous summarizer. The interface explicitly warns about earlier-court confusion and provides original pages, final-page navigation and separate reviewer edits. These UI changes are not a model-quality improvement on this test.

For the next model revision, treat these now-inspected cases as development evidence, obtain paragraph/decision-span annotations, and reserve a new untouched test set before tuning. Include difficult quotations, connected appeals, conditional relief, multiple orders and scanned documents.
