# Faculty walkthrough — meeting preparation

## Your opening explanation (about 30 seconds)

“My mini project is an English legal judgment summarizer. I first trained models to identify the role of each sentence, such as facts, arguments, reasoning or ruling. Then I trained a separate selector to choose important sentences. The final app shows a structured extractive brief with links to the original PDF, so a person can verify the result.”

## A 10–12 minute presentation

| Time | Show | Explain |
|---|---|---|
| 1 minute | Architecture diagram | Sentence roles and sentence importance are different tasks |
| 2 minutes | Notebook 01: sample rows and audit | Preserve case boundaries, label policy, no train/dev document overlap |
| 3 minutes | Notebook 02: model diagram and weak-label example | TF-IDF → InLegalBERT → document BiLSTM → selector; no reference at inference |
| 2 minutes | Notebook 03: recorded comparisons | Macro F1, ROUGE, controls and honest limitations |
| 2–3 minutes | Notebook 04 or local app | Upload, source reference, final-page check, reviewer draft/export |

Do not retrain models live in the meeting. Run the demo beforehand and keep `examples/sample_summary.md` available if the runtime disconnects. Keep `RUN_TRAINING=False`. If you demonstrate evidence-only mode, say explicitly that trained ranking is disabled; for the actual model, download the frozen release first and enable it.

If your faculty wants to inspect the initial ML pipeline, open [Notebook 00](../notebooks/00_classical_ml_and_word_bilstm.ipynb). It shows TF-IDF, majority, KNN, logistic regression and SVM inline, with optional word-BiLSTM training. Keep `RUN_WORD_BILSTM=False` for the quick demonstration. [The code map](code_map.md) explains which functions the short main notebooks call and where their implementations live.

## Questions you are likely to get

**Why not only TF-IDF?** It is a useful baseline but mostly captures word overlap. BERT provides contextual language representations; the experiments measure whether that helps our role task.

**What did you train yourself?** The classification fine-tuning, BiLSTM role heads and importance selectors. InLegalBERT pretraining belongs to the original authors. Do not say you built a 110-million-parameter language model from scratch.

**What are the two BiLSTMs?** Stage 1 reads words inside one sentence. The later document BiLSTM reads BERT sentence vectors across a judgment. They operate at different sequence levels.

**Why CRF? Is it in the app?** CRF models neighboring label transitions and chooses a joint role sequence. It was an optional comparison. The current deployed head uses softmax; CRF did not improve macro F1 in the recorded comparison.

**How did you label important sentences?** Heuristically from training reference summaries, using TF-IDF similarity or marginal reference n-gram coverage under the same 400-word budget. These are weak labels, not human sentence judgments.

**Did roles improve summaries?** Not consistently. The controlled results show small mixed changes. A no-role selector is a valid control and remains one deployed option. Report the evidence rather than assuming the proposed idea must win.

**Why is macro F1 lower than accuracy?** Frequent roles dominate accuracy. Macro F1 treats rare roles equally, exposing weaknesses that accuracy can hide.

**Is 0.52 ROUGE-1 “52% accuracy”?** No. It is word-overlap F1 against reference summaries in the recorded protocol. It does not establish correct legal meaning or outcome.

**Can it handle long judgments?** The app scores all extracted sentences in overlapping windows. BERT still truncates a long target at 192 tokens, and extraction/attribution can fail. Coverage does not mean completeness.

**Can an extractive summary hallucinate?** It avoids free-form invented wording, but can mislead by taking true source text from the wrong context, such as an earlier ruling. Our separate review found that failure, so a person checks the source.

**What is new in this repository?** Four coherent notebooks, reusable code, documented provenance, reproducible data/model access, recorded experiment tables, tests and a review interface. Consolidation itself is not a new model-quality improvement.

**What would you improve next?** Better current-court attribution and decision-span labels, explicit abstention, and independently reviewed unseen cases. Larger models or more rules alone do not prove improvement.

## Vocabulary in one line

- Embedding: a numerical representation of text.
- Fine-tuning: adapting a pretrained model to a labeled task.
- Logits: raw prediction scores before probabilities.
- Cross-entropy: a loss penalizing wrong class probabilities.
- Backpropagation: computing how each parameter contributed to loss.
- Epoch: one pass through training examples.
- Early stopping: stop when development performance stops improving.
- Ablation: change one component while keeping the comparison controlled.
- Extractive summary: selected source sentences.
- Weak supervision: labels created by a heuristic rather than direct human annotation.
- Evidence span: exact source location supporting an extract.

## Questions to ask your faculty

1. Should the next milestone emphasize role classification, summary quality, or the end-to-end demo?
2. Can we obtain independent legal review or annotated current-decision spans?
3. What evaluation size and baseline comparisons are expected for the mini project?

## Current automatic summarization result

The current 4.6.0 pipeline at its default 800-word target achieved **20.94 ROUGE-L F1 points**, with a 95% case-bootstrap interval of **19.41–22.58**, on all **100 official IN-Abs test judgments**.

The current structured pipeline scores below the opening-sentence baseline on ROUGE-L. These results do not establish a reference-overlap improvement from the pipeline. The decision fixes addressed specific attribution errors; they did not establish better overall summarization.

Report this as ROUGE-L F1, not legal accuracy. [Full results and presentation wording](current_summary_benchmark.md).
