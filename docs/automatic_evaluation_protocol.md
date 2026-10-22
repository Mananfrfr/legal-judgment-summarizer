# Automatic evaluation of the current summarizer

This evaluation measures **reference-summary overlap**, not the percentage of legally correct summaries. It uses the frozen 4.6.0 pipeline, unchanged trained weights and all 100 official IN-Abs test cases. No score threshold is invented to turn overlap into pass/fail accuracy.

## The comparison

- Primary result: macro-average **ROUGE-L F1** for the app's default 800-word setting.
- Supporting results: ROUGE-1 and ROUGE-2 precision, recall and F1; actual summary lengths; missing-section and exact-source checks.
- Baseline: opening consecutive sentences from the same parsed judgment, at the same requested budget.
- Secondary comparison: both methods at 400 words. This helps interpret length effects but does not reproduce the old 120-candidate Stage 3.6 experiment.
- Uncertainty: 10,000 case-level bootstrap resamples, fixed seed 42, percentile 95% confidence intervals. Differences use paired case resampling.
- Failure handling: retain every case in the denominator. A failed method output receives zero overlap scores, and failures are reported explicitly.

The app uses a soft word budget to preserve complete sentences and operative conditions. It may exceed the requested length. The baseline can exceed its budget only for an overlong first sentence. Actual lengths and over-budget counts must accompany the scores; the methods are not guaranteed equal output lengths.

The evaluation concatenates only the displayed background, issue, reasoning and decision extracts, in app order. It excludes headings, source citations, audit warnings and suggested omitted passages. It does not reorder the app's output to improve ROUGE-L.

## How contamination is handled

The test set was already evaluated in Stage 3.6. This is a **reused benchmark**, not a fresh blind sample. The inference code, weights, source/reference hashes, selected methods and metrics were frozen before this run. Reference summaries are not supplied to the model or selection rules. No pipeline tuning is performed in response to these scores.

The exact-text audit checks all 7,030 training judgment files, including the two without usable paired summaries, against the 100 test sources. It also identifies the recorded 1,000 training and 40 development IDs used for selector training. This does not prove absence of semantic duplicates or upstream role-encoder/pretraining exposure.

These are publisher TXT files. This benchmark does not measure OCR, PDF layout recovery or generalization to different courts, languages, time periods or document formats. Bootstrap intervals describe variation across these cases; they do not account for those broader uncertainties.

## Reproduce a new run

Install the project's training dependencies and a compatible PyTorch runtime, then download the verified model assets and official dataset:

```bash
python scripts/download_models.py
python scripts/download_data.py inabs
python scripts/evaluate_current_summarizer.py prepare
python scripts/evaluate_current_summarizer.py generate --device cpu
python scripts/evaluate_current_summarizer.py report
```

Use `--device cuda` with a compatible NVIDIA runtime for faster generation. The recorded run uses Torch 2.6.0+cu124, CUDA 12.4, an RTX 2050, batch size 3 and the production model's CUDA autocast path. The local app was checked on Torch 2.14.1 CPU; numerical equivalence across these runtimes has not been established. The evaluator records actual versions and refuses to mix runtimes when resuming.

For weights in another location, add `--encoder PATH --bundle PATH` to `prepare` and `generate`. Use a new `--output` folder for each new experiment. `generate` can resume matching completed cases; `report` refuses to score an incomplete 100-case run. Original source texts, generated summaries and model predictions stay under ignored `outputs/`. Public reports contain metrics, IDs and checksums, not a mirrored corpus.

`prepare` hashes code and weights before generation. A preflight protocol was superseded before any predictions were generated to expand the leakage audit from 7,028 paired training cases to all 7,030 source judgments. The published protocol is the one used for the entire scored run.

## Sources

The [official dataset](https://zenodo.org/records/7152317) provides IN-Abs judgment/reference pairs. Metrics use [Google's rouge-score implementation](https://github.com/google-research/google-research/tree/master/rouge), with stemming and sequence-level ROUGE-L. The distinction between automatic metrics and human judgments is central to [SummEval](https://aclanthology.org/2021.tacl-1.24/). GPU installation follows the [official PyTorch version instructions](https://pytorch.org/get-started/previous-versions/).
