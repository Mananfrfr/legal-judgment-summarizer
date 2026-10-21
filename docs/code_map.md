# Why the main notebooks are short

The project was reorganized into **four main notebooks plus an optional detailed Stage 1 notebook**. Short notebook cells call reusable Python functions. For example, `train_baselines(...)` in Notebook 02 performs TF-IDF fitting, trains four classifiers, calculates metrics and saves models inside `src/train_models.py`. It does not replace those steps with a different algorithm.

Notebook 00 now shows these classical training steps inline, followed by the optional word-level BiLSTM architecture and training loop. Use it when the faculty wants to inspect the early ML experiments rather than only the end-to-end overview.

## Where to find each part

| Part | Readable walkthrough | Full implementation or history |
|---|---|---|
| Majority, KNN, logistic regression, SVM | [Notebook 00](../notebooks/00_classical_ml_and_word_bilstm.ipynb), steps 1–6 | Inline in 00; shared entrypoint in [train_models.py](../src/train_models.py) |
| Stage 1 word-level BiLSTM | Notebook 00, steps 7–9 | Inline architecture, input packing, weighted loss, training and early stopping; [original Stage 1 source](../research/legal_summarizer_stage1_107VMw.py.txt) |
| Dataset parsing and annotation audit | [Notebook 01](../notebooks/01_data_preprocessing.ipynb) | [annotations.py](../src/annotations.py), [data_utils.py](../src/data_utils.py) |
| InLegalBERT fine-tuning and document BiLSTM | [Notebook 02](../notebooks/02_model_training.ipynb) | [train_models.py](../src/train_models.py) |
| Optional CRF comparison | Notebook 02 | [crf.py](../src/crf.py), [train_models.py](../src/train_models.py) |
| Weak sentence labels and four selector experiments | Notebook 02 | [selection_training.py](../src/selection_training.py), [train_selector.py](../src/train_selector.py), [dataset_index.py](../src/dataset_index.py) |
| Historical performance and errors | [Notebook 03](../notebooks/03_model_evaluation.ipynb) | [evaluation](../evaluation), [methodology](methodology.md) |
| Full-document inference | [Notebook 04](../notebooks/04_final_pipeline.ipynb) | [project_api.py](../src/project_api.py), [stage44_model.py](../src/stage44_model.py), [stage44_windows.py](../src/stage44_windows.py) |
| PDF text, evidence, attribution and brief rules | Notebook 04 | [stage44_source.py](../src/stage44_source.py), [stage44_attribution.py](../src/stage44_attribution.py), [stage44_brief.py](../src/stage44_brief.py) |
| Review app | Notebook 04, final section | [stage44_app.py](../src/stage44_app.py), [review_server.py](../src/review_server.py), [review_service.py](../src/review_service.py) |
| Older experiments, including alternative losses | [methodology](methodology.md) | [research source exports](../research), linked to the original Colabs in [source inventory](source_inventory.json) |
| Trained parameters | [models README](../models/README.md) | [Versioned GitHub release](https://github.com/Mananfrfr/legal-judgment-summarizer/releases/tag/v0.1.0-faculty) |

## Code, weights and data are different

- **Code** specifies the algorithm. It lives in the notebooks, `src/`, `scripts/` and tests.
- **Weights** are the learned numerical parameters. The encoder archive is about 406 MB; the selector/head bundle is about 4.3 MB. They are release downloads with checksums, so they do not make Git history unnecessarily large.
- **Datasets** are obtained from the original publishers through [data access instructions](../data/README.md). The repository includes fixed split IDs and download/checksum logic. It does not rehost the gated role corpus.
- **Research exports** preserve historical code and paths; they are reference text, not ready-to-run replacements for the current notebooks. The main supported workflow covers the principal experiments, not every historical variation as a separate runnable notebook.

The deployed inference modules and model weights remain unchanged. The new Stage 1 notebook is a readable training workflow, not a newly improved deployed model or proof that historical scores have been reproduced.
