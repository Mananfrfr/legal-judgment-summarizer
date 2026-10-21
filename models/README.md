# Frozen model artifacts

Run `python scripts/download_models.py` to download the versioned GitHub Release assets and verify every SHA-256 digest from `release_assets.json`.

- `encoder.zip` contains the existing fine-tuned context256 InLegalBERT folder.
- `stage4_english_model_bundle.zip` contains the existing softmax BiLSTM role head, selector checkpoints and compatibility manifest.

The inference loader checks the encoder fingerprint, head archive digest, selector digests and state shapes. These weights were **not retrained or improved during notebook consolidation**. Newly trained research outputs need independent evaluation and a new matching deployment bundle.

The encoder derives from [Law-AI InLegalBERT](https://huggingface.co/law-ai/InLegalBERT), whose model card lists MIT licensing. See `THIRD_PARTY_NOTICES.md` for attribution. The project does not claim to have pretrained InLegalBERT from scratch.

Original Drive locations are retained in `manifest.json` for provenance. Drive links may require permission; GitHub Release assets are the public download route. Model files are ignored by Git so clones remain small.
