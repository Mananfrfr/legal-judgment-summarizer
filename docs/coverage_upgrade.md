# Version 4.7: improve coverage without changing the trained weights

The frozen InLegalBERT, document BiLSTM and importance selector are unchanged. The changes are in the code that extracts and selects the model's evidence. The old pipeline often discarded useful sentences after scoring them.

## What changed, in simple terms

1. **Quotation marks no longer blindly toggle quotation state.** A stray closing straight quote used to make the remaining judgment look like a quotation. The parser now examines neighboring punctuation to distinguish opening and closing marks. Genuine quoted orders remain excluded; uncertain or missing quotation boundaries still need review.
2. **Recognize more ways courts state the outcome.** Short closing forms such as “Appeal allowed,” “we accept the appeal,” and a late “this appeal must be dismissed” can establish a decision. Negation, reported orders, quotations and submissions remain guarded. Recognizing an ending does not prove that it is the majority disposition in a multi-opinion judgment.
3. **Use the requested space more effectively.** After selecting the core sections, spend unused word allowance on additional source sentences ranked by learned importance, position and a redundancy penalty. The previous three-sentence background cap no longer wastes the remaining allowance. Additional sentences must fit the remaining budget; existing complete operative passages may still overflow.
4. **Keep source attribution visible.** Known submissions, letters and earlier rulings cannot be promoted into current reasoning by a high model score. Unresolved sentences can enter reasoning when the predicted ANALYSIS/RATIO role score is at least 0.65; this is an uncalibrated heuristic and receives an explicit attribution-review warning.
5. **Handle broken paragraph boundaries.** A huge source paragraph can contain hundreds of sentences or separate opinions. The decision extractor now keeps local operative sentences and adjacent conditions from such a paragraph instead of copying the entire block. It warns that the paragraph boundary and separate opinions require checking.

Every displayed extract still carries source spans. No sentence is paraphrased and no missing outcome is invented. Exact copying verifies evidence integrity; it does not establish legal correctness or completeness.

## How the candidate was selected

Development used the existing 40 IN-Abs development cases. The initial coverage candidate exposed a giant-paragraph failure; that was fixed before freezing the final candidate. On these development cases, 800-word ROUGE-1/2/L F1 changed from **38.22 / 19.62 / 21.92** to **56.68 / 35.38 / 31.69**. Lead-800 scored **26.89 ROUGE-L**. Empty decision sections fell from **17 to 4**. These are tuning results, not held-out evidence.

A separate sample of 100 paired judgments was reserved before tuning, with seed 20261022. It comes from the unused portion of IN-Abs's published training partition. The exclusion audit read the Stage 3.2–3.6 manifests: their combined known train/dev IDs total 1,040. Exact normalized source duplicates of those IDs and the previously inspected official test set were excluded, and the reserved sample contains no exact duplicate sources. Model pretraining overlap, semantic near duplicates and unrecorded exposure are not cleared. This is a same-corpus reserved evaluation, not an external or independently human-validated test.

The evaluator freezes all Python source files, its own code, the protocol, the encoder and selector archive, and generates every method's outputs before comparing them with references. One **pre-review safety amendment** followed the initial freeze: a synthetic test caught an opening straight quote immediately after a colon being classified as closing punctuation. The initial reserved report had been computed but had not been read. The one-character fix and regression test were completed before inspecting reserved metrics, the previous freeze/output were preserved, and the final code was frozen and scored again. The amendment record is published with both freezes. This is not a claim of a single unmodified first run, and there was no metric-driven reserved-set tuning. The original 4.6 benchmark and its 20.94 ROUGE-L result remain unchanged.

## Measurement

| Method (800-word target) | ROUGE-1 | ROUGE-2 | ROUGE-L | Empty decisions |
|---|---:|---:|---:|---:|
| Previous 4.6 pipeline | 41.00 | 19.98 | 21.98 | 45 |
| Opening-sentence baseline | 50.47 | 25.01 | 24.81 | N/A |
| Upgraded 4.7 pipeline | 56.67 | 33.62 | 30.23 | 13 |

The paired 800-word ROUGE-L improvement over the previous pipeline is 8.25 points (95% CI 6.40–10.09); over lead it is 5.42 points (95% CI 3.98–6.95).

See `evaluation/coverage_upgrade_reserved.json` for the reserved result, `coverage_upgrade_per_case.csv` for every case/method score, and `coverage_upgrade_freeze.json` / `coverage_upgrade_protocol.json` for provenance. Compare old and new pipelines **on the same rows**, not a new number against an older result on different documents.

The primary comparison is structured, displayed-order ROUGE-L F1 at an 800-word target. Report ROUGE-1/2, a lead baseline, the 400-word comparison, output lengths, overflows, empty sections and source checks as well. Intervals use 10,000 case-bootstrap resamples with seed 42; differences use paired resampling. ROUGE scores are points, not percentages of legally correct summaries. Empty-section reduction measures availability, not correct attribution of every recovered section.

The long-document evaluator uses exact bit-parallel longest-common-subsequence computation. Tokenization, Porter stemming and ROUGE-1/2 use `rouge-score`. Tests compare every score with the original implementation on 100 randomized sentence pairs plus empty, reordered, repeated and stemmed examples. This is a speed improvement, not a new metric.

At the secondary 400-word target, ROUGE-L is **28.04**, versus **21.10** for 4.6 and **21.01** for lead. Nine upgraded outputs exceed that soft target to retain complete passages; at 800 words none of the 100 reserved outputs exceeds the target. The 800-word improvement partly reflects better use of the available space (782.6 words versus 306.4), so it should not be described as an isolated gain in the trained model's accuracy.

Validation: **104 automated tests passed**; both changed notebooks (03 and 04) executed through their isolated kernels. The three earlier failure PDFs retained their targeted decision checks, the original ten PDFs retained 29/29 content and 15/15 review checks, and the four additional development PDFs retained all specified decision phrases. Changed sentence segmentation was rescored where old caches no longer matched. These phrase checks are regressions on inspected documents, not independent legal correctness labels. The evaluation and PDF checks include CUDA inference; local app inference uses its existing CPU runtime.

## Reproduce

Use a CUDA-capable environment, the documented dataset archive and the original release weights. The locally tested evaluation runtime is Torch 2.6.0+cu124 on an RTX 2050. Runtime equivalence with other Torch/CPU environments is not established. Download weights into `models/` and the dataset into `data/raw/`. Git history must contain baseline commit `6071e3c64f92199edd972c81f408bf5c0c1b2664`.

```bash
python scripts/evaluate_coverage_upgrade.py prepare
python scripts/evaluate_coverage_upgrade.py generate --split development
python scripts/evaluate_coverage_upgrade.py score --split development --label reproduced
python scripts/evaluate_coverage_upgrade.py freeze
python scripts/evaluate_coverage_upgrade.py generate --split reserved
python scripts/evaluate_coverage_upgrade.py score --split reserved --label reproduced
```

Preparation reads original Stage 3 result archives from Downloads when available; otherwise it uses the published exposure audit. It snapshots the original baseline from Git. Model caches can only be reused when the source and sentence sequence match; changed segmentation is scored again. Frozen runs reject modified code or checkpoints. Failed/incomplete runs do not produce a completed aggregate; do not silently drop their cases. Keep new reruns separate from the published report.

## Remaining limits

This is still a review-assisted research app. More coverage can introduce distracting context; role predictions and textual attribution rules can be wrong. Quotation corruption, separate opinions, OCR, pronouns and implicit outcomes remain difficult. The reserved sample measures TXT reference overlap, not scanned-PDF quality. Production legal reliability requires independently annotated decision/attribution checks and evaluation on external courts, years and formats.
