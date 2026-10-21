"""Entrypoint for the review-assisted app."""
import os
from pathlib import Path
from review_server import main

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('LEGAL_ENCODER_DIR',str(ROOT/'models/encoder'))
os.environ.setdefault('LEGAL_MODEL_BUNDLE',str(ROOT/'models/stage4_english_model_bundle.zip'))
os.environ.setdefault('LEGAL_REVIEW_DATA',str(ROOT/'.runtime'))
if __name__=="__main__":main()
