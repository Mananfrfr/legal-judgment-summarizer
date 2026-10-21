# Consolidation validation — 21 October 2026

**22 October candidate update:** the `improve-decision-attribution` branch changes two inference modules and passes 76 tests plus development-document checks. The statements below describe the original consolidation and released Stage 4.5 code. See [the separate candidate report](decision_upgrade.md) for the new revision; its weights are unchanged and its unseen-document accuracy is unmeasured.

## Verified

- 71 automated tests pass: the original 62 pipeline/app checks plus nine teaching-workflow checks. The additions cover document-split leakage, baseline fitting, no-overwrite protection, same-document context packing, padding masks, CRF versus brute-force calculation, budgeted weak labels, real one-epoch fixture training for softmax/CRF heads and all four selector conditions.
- All five notebooks (four main notebooks and the optional detailed Stage 1 notebook) execute in default demo mode through the isolated-kernel mechanism. Recorded metric tables and figures render; synthetic data and scores are labeled. The demo pipeline creates a source-checked summary and JSON export.
- The optional Stage 1 notebook also executes with word-BiLSTM training enabled for one synthetic epoch, including checkpoint save/reload. Additional checks confirm identical classical metrics to the shared training helper, padding-invariant word-model predictions, and unknown/empty word handling. This is software verification, not an official-data training run.
- The six core inference modules (source, attribution, windows, model, brief, CLI) are byte-for-byte identical to Stage 4.5. No checkpoint or inference rule was changed to make the presentation simpler.
- Original notebook stages were discovered in the connected Drive, downloaded, inventoried and reviewed. Python-source text exports preserve the project history; the four new notebooks form the supported narrative.
- Recorded role and summary tables are derived from saved Colab outputs/result archives with source provenance. They are not fresh model-quality measurements.

## Not claimed

Full official-dataset training was not rerun. Training helpers have fixture coverage but are not a verified numerical reproduction of every historical experiment. New model weights require fresh evaluation and a compatible provenance bundle before deployment. Hosted Colab dependency installation/GPU execution was not tested in a real hosted session in this consolidation; local notebook execution used Python 3.11 and Torch 2.14.1+cpu. Colab setup specifies Torch 2.8.0/cu128 as in the prior notebook workflow.

The original current-pipeline limitation remains: two of three separate judgments had serious decision-section failures, even while selected evidence matched source text. No independent human legal accuracy, broad OCR evaluation or autonomous production readiness is established. See `current_pipeline_evaluation.md`.

The public repository contains code, documentation, recorded metrics, manifests, fixed split IDs and safe invented examples. Publisher datasets are downloaded at their official sources; trained model artifacts are versioned release downloads rather than large Git blobs. No credentials, user uploads, local environments or cached embeddings are intentionally published.
