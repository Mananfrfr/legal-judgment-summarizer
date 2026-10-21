# Original source exports

These text files preserve code from the relevant Stage 1–4 Colab experiments, including the staged training expansions and alternative losses. Source notebook links and SHA-256 fingerprints are in `docs/source_inventory.json`. Outputs and any recognized embedded Hugging Face token strings were removed.

They are **reference material**, not the four main notebooks and not directly executable scripts. They preserve historical cell-level state, paths and flags, including expensive training and the already-completed official test. Some exports contain embedded file-writing cells. Do not copy them wholesale into a running session or rerun the official test during model selection.

Use `notebooks/01...04` and the named `src/` functions for the supported workflow. Read the source exports when explaining a historical experiment or checking how the original results were produced.

For the initial ML experiments, [Notebook 00](../notebooks/00_classical_ml_and_word_bilstm.ipynb) now provides a runnable, step-by-step version of the classical baselines and optional word BiLSTM. The historical source remains here for provenance.
