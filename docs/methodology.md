# Methodology and experiment decisions

## 1. Problem and scope

Build an English extractive legal judgment summarizer. It selects source text and organizes a brief; it does not generate an unrestricted paraphrase or predict a legal outcome. Source copying can still mislead when it attributes an earlier ruling to the current court, so attribution and review are necessary.

## 2. Classification experiments

The role corpus has 13 labels. Preserve offsets and document boundaries; use the original split and annotation policy. Fit TF-IDF only on training data. Compare majority, five-neighbor KNN, balanced logistic regression and balanced linear SVM. The Stage 1 word BiLSTM is a separate baseline, not the final document-context architecture.

Fine-tune `law-ai/InLegalBERT` for 13 labels. Stage 2.2 compares target-only and neighboring-context inputs under a 256-token limit. Context mode reserves up to 192 tokens for the target, then fills available space with adjacent text from the same judgment. Use three epochs, AdamW learning rate 2e-5, weight decay .01, warmup .1, batch 8 and accumulation 4. Select on development macro F1.

Freeze the fine-tuned encoder. A bidirectional LSTM reads sentence embeddings in document order (128 hidden units per direction, dropout .3). Compare softmax and an actual linear-chain CRF with masked likelihood and Viterbi decoding. Train with AdamW .001, document batch 4, at most 25 epochs and patience 5. The softmax head had slightly better recorded macro F1; the CRF had very slightly better weighted F1. Later weighted-loss experiments did not beat the original macro F1. Do not claim that extra complexity automatically improved performance.

## 3. Summary selection experiments

Stage 3 evolved from a role-guided MVP and IN-Ext exploration through 60/15 and 200/40 train/dev pilots, then 1,000/40 cases. Stage 3.5 compared coverage/weak labels; Stage 3.6 corrected the comparison to use a matched 400-word teacher budget. The primary recorded final comparison is Stage 3.6, not an aggregation of differently sized pilot runs.

Two weak teachers use training reference summaries: TF-IDF cosine similarity and greedy marginal unigram/bigram reference coverage. They create binary importance labels, not expert annotations. Four selectors compare the two teachers with predicted-role features on/off. The MLP is LayerNorm → Linear(160) → ReLU → Dropout(.25) → Linear(1); inputs have 769 or 782 features. Train with weighted binary cross-entropy, AdamW 8e-4, batch 128, 16 epochs and patience 4. Select checkpoints on development ROUGE-L. Extraction rejects high TF-IDF redundancy (threshold .86), obeys a 400-word whole-sentence budget and restores source order.

The historical benchmark takes 60 beginning, 30 middle and 30 ending candidate sentences for sufficiently long judgments. It is not full-document inference. Recorded test comparison: 100 official IN-Abs cases, all six frozen methods, once. The saved manifest and paired comparisons are in `evaluation/recorded/stage36`.

## 4. Current inference pipeline

Reuse verified trained weights. Extract page-addressable sentences; score all candidates with overlapping windows; aggregate repeated scores; organize four source-extract sections; run exact evidence and omission checks. Flag uncertain extraction, quotation recovery, missing sections and long target truncation. Some directives can exceed the word target to keep their conditions together.

The original consolidation copied the Stage 4.5 inference modules. The subsequent [4.6 revision](decision_upgrade.md) changes parser quotation recovery, attribution and decision assembly; windows, model weights and model scoring remain unchanged. Its measured gains are targeted development checks, not a new summary-accuracy estimate. The notebook training interfaces are refactored research entrypoints with the same principal architectures/settings, not a new numerical reproduction of every historical experiment. Original source exports are kept for detailed comparison. Newly trained weights require new evaluation and a provenance-compatible deployment bundle; they are not automatically promoted.

## 5. Evidence and limitations

Role metrics are on DEV. Stage 3.6 ROUGE is on its original official TEST protocol. Neither is an accuracy score for today's full-document app. A separate three-judgment source comparison found two critical decision-section failures; all selected spans could still match source text exactly. No independent human legal-accuracy benchmark has been completed, and training overlap for those three judgments is unknown.

Future work: label current-court and reported/quoted spans, evaluate attribution before selection, calibrate abstention, validate scanned/native formats and conditional orders, then test on a newly reserved set with independent reviewers. Changing rules to fix inspected examples requires a new untouched evaluation set.
