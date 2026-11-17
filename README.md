# Legal Judgment Summarizer

A local app for turning English-language Indian judgments into extractive case briefs. Upload a PDF or text file to get a brief covering **background, issue, reasoning and decision**. Select a passage in the brief to see it highlighted on the original page. You can save a reviewer draft and export it alongside the machine summary and its evidence.

## Quick start

The following setup uses Python 3.11 and CPU inference. From the repository root, create a virtual environment and activate it in your shell.

```sh
python -m venv .venv
```

On Windows PowerShell, run `.\.venv\Scripts\Activate.ps1`. On Linux, run `source .venv/bin/activate`. Then install the app and download the trained checkpoints.

```sh
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-app.txt
python scripts/download_models.py
python src/stage44_app.py
```

Open the [local app](http://127.0.0.1:7861/) and choose **Open example judgment** to try it. The downloader verifies the encoder and selector bundle against their release checksums. Searchable PDFs and UTF-8 text files work directly. For scanned PDFs, install Tesseract and enable OCR. [Model files and provenance](models/README.md)

## How it works

1. **Extract the text.** The pipeline splits the judgment into sentences and keeps the page and character offsets for each extract.
2. **Score the sentences.** InLegalBERT encodes each sentence in context. A document-level BiLSTM predicts rhetorical roles such as facts, arguments and rulings. A trained selector scores sentences across overlapping windows so it can reach orders near the end of a long judgment.
3. **Build and review the brief.** The pipeline groups selected passages into the four sections and checks their source spans. In **Parsed details**, you can compare outcomes, rupee amounts and deadlines with their source passages and inspect possible omissions. [Architecture](docs/architecture.md) · [Parsing rules](docs/summary_correctness_checks.md)

### Optional local LLM judge

On Windows x64, run `.\Start-Judge.ps1 -Install` in a second PowerShell window. The first run downloads about 1.15 GB for Qwen2.5-1.5B-Instruct and llama.cpp. On later runs, use `.\Start-Judge.ps1`. Both the app and the judge process documents locally.

In **LLM review**, select **Run local judge**. It checks the last decision extract and the first extract in each other populated section, up to four extracts in total. It returns *supported*, *contradicted* or *insufficient evidence*. Open the cited page to check its interpretation. In a controlled example, it caught a changed amount and deadline but missed a wrong court outcome. [Judge validation and examples](docs/local_judge_validation.md)

## Evaluation

### Summarization

The 4.7 pipeline was evaluated on **100 reserved IN-Abs judgments** with an **800-word target**. Precision measures how much of the generated wording overlaps the reference summary. Recall measures how much of the reference wording the generated summary covers. F1 combines the two. Each score is computed per judgment and then averaged, on a 0–100 scale.

| Reference-overlap metric | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| ROUGE-1 (single words) | 53.54 | 69.93 | 56.67 |
| ROUGE-2 (word pairs) | 31.72 | 41.88 | 33.62 |
| ROUGE-L (longest common subsequence) | 28.30 | 38.01 | 30.23 |

On the same judgments, the opening-sentence baseline scored **24.81 ROUGE-L F1** and the previous pipeline scored **21.98**. The current pipeline's **30.23** has a 95% case-bootstrap interval of **28.57–31.96**. All 100 briefs passed the source-span checks, but 13 had an empty decision section. A passage can be copied exactly and still describe the wrong court's ruling. Whole-summary legal correctness remains unmeasured. [Per-case results](evaluation/coverage_upgrade_per_case.csv) · [Evaluation protocol and limitations](docs/coverage_upgrade.md)

### Sentence-role classification

This experiment measures how well each model assigns one of 13 rhetorical roles to a sentence on the OpenNyAI development split. Macro F1 gives each role equal weight. Weighted F1 gives more weight to roles with more examples.

| Model | Accuracy (%) | Macro F1 (%) | Weighted F1 (%) |
| --- | ---: | ---: | ---: |
| TF-IDF + logistic regression | 46.61 | 39.21 | 48.06 |
| Word BiLSTM | 50.05 | 41.35 | 50.82 |
| Context-aware InLegalBERT | 76.31 | 61.49 | 75.69 |
| InLegalBERT + document BiLSTM | 83.12 | 65.41 | 82.37 |

The saved classification results contain accuracy and F1. Computing per-role precision and recall requires the predicted and gold sentence labels. [All role-classification experiments](evaluation/role_results.csv) · [Detailed metrics report](docs/faculty_metrics_report.md)

## Data and notebooks

OpenNyAI InRhetoricalRoles supplies sentence labels for role classification. IN-Abs supplies judgment–summary pairs for selector development and evaluation. Dataset access instructions, checksums and split IDs are in the [data guide](data/README.md).

| Notebook | Contents |
| --- | --- |
| [00 · Classical ML and word BiLSTM](notebooks/00_classical_ml_and_word_bilstm.ipynb) ([Colab](https://colab.research.google.com/github/Mananfrfr/legal-judgment-summarizer/blob/main/notebooks/00_classical_ml_and_word_bilstm.ipynb)) | TF-IDF, KNN, logistic regression, SVM and a word-level neural baseline |
| [01 · Data preprocessing](notebooks/01_data_preprocessing.ipynb) ([Colab](https://colab.research.google.com/github/Mananfrfr/legal-judgment-summarizer/blob/main/notebooks/01_data_preprocessing.ipynb)) | Annotation handling, document splits and leakage checks |
| [02 · Model training](notebooks/02_model_training.ipynb) ([Colab](https://colab.research.google.com/github/Mananfrfr/legal-judgment-summarizer/blob/main/notebooks/02_model_training.ipynb)) | InLegalBERT, document-level role prediction and sentence selection |
| [03 · Model evaluation](notebooks/03_model_evaluation.ipynb) ([Colab](https://colab.research.google.com/github/Mananfrfr/legal-judgment-summarizer/blob/main/notebooks/03_model_evaluation.ipynb)) | Classification metrics, reference-summary comparisons and error analysis |
| [04 · Final pipeline](notebooks/04_final_pipeline.ipynb) ([Colab](https://colab.research.google.com/github/Mananfrfr/legal-judgment-summarizer/blob/main/notebooks/04_final_pipeline.ipynb)) | Generate a brief from a new document and inspect its evidence |

## Reproduce the checks

To rebuild the metrics report from the saved evaluation records, run `python scripts/build_faculty_metrics_report.py`. Run the test suite in the activated environment with the following commands.

```sh
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
```

[Methodology](docs/methodology.md) · [Faculty walkthrough](docs/faculty_walkthrough.md) · [MIT license](LICENSE) · [Third-party notices](THIRD_PARTY_NOTICES.md)
