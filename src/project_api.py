"""Notebook-friendly entrypoints around the evidence-grounded inference pipeline."""
from pathlib import Path
import json,os

def load_deployed_model(root):
    from stage44_model import FullDocumentRanker
    import torch
    torch.set_num_threads(4)
    root=Path(root)
    encoder=Path(os.environ.get('LEGAL_ENCODER_DIR',root/'models/encoder'))
    bundle=Path(os.environ.get('LEGAL_MODEL_BUNDLE',root/'models/stage4_english_model_bundle.zip'))
    if not (encoder/'model.safetensors').exists() or not bundle.exists():
        raise FileNotFoundError('Download the trained model release with scripts/download_models.py, or set LEGAL_ENCODER_DIR and LEGAL_MODEL_BUNDLE.')
    return FullDocumentRanker(encoder,bundle)

def summarize_document(path,model=None,budget=800,ocr='off'):
    """model=None is explicitly evidence-only; never silently fall back on failure."""
    from stage44_brief import summarize_file
    return summarize_file(path,budget,model,ocr=ocr)

def save_summary(result,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    (folder/'summary.md').write_text(result['markdown'],encoding='utf8')
    (folder/'evidence.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8')
    return folder
