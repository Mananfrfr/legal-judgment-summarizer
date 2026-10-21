# Data access and split policy

| Data | Purpose | How to obtain |
|---|---|---|
| OpenNyAI InRhetoricalRoles | Supervised 13-role sentence classification | Accept [publisher access conditions](https://huggingface.co/datasets/opennyaiorg/InRhetoricalRoles), then run `python scripts/download_data.py roles` with an authorized HF login/token |
| IN-Abs | Reference summaries for weak-label training and ROUGE evaluation | Run `python scripts/download_data.py inabs` to obtain the [official ZIP](https://zenodo.org/records/7152317) |
| IN-Ext | Earlier expert-reference extraction exploration and overlap checks | Included in the same official ZIP; do not merge it into training without overlap auditing |
| Synthetic teaching example | Offline notebook and app demonstration | `examples/demo_judgment.txt`; invented, not a legal case |
| Separate judgment checks | Current-pipeline qualitative error analysis | URLs and hashes in the evaluation files; these cases are now inspected, not a fresh test set for further tuning |

The Git repository contains dataset instructions, revision/checksum metadata and fixed split IDs. It does **not** contain the gated role corpus, the original 97.8 MB IN-Abs archive, all judgment texts, uploaded files or cached embeddings. This keeps the public repository small and preserves publisher access conditions.

Role revision: `fc6664bae469e6c47acf253704622d593dc7465e`.
Official summarization archive MD5: `c77948fe580d26200f5c9fe53b81485d`.
`splits.json` contains the historical 1,000 training and 40 development IN-Abs case IDs. The recorded 100-case official test was already completed; keep its results fixed. Do not generate training labels from development/test summaries. `data/raw` and `data/processed` are ignored by Git.

An alternative to HF download is to place the three original role files at `data/raw/roles/train.json`, `dev.json` and `test.json`, then call `load_role_data(..., download=False)`. Annotation selection prefers an explicit adjudication marker, otherwise the first nonempty annotation set; disagreements are reported. Exact document duplicates across splits cause a failure.
