"""Small public interface for audited role data and a clearly labeled teaching demo."""
from pathlib import Path
import json
from annotations import LABELS, json_to_df, document_hashes

REVISION='fc6664bae469e6c47acf253704622d593dc7465e'
FILES={'train':'data/train-00000-of-00001-bb0092e0d8549337.parquet',
       'dev':'data/dev-00000-of-00001-af55705c75623915.parquet',
       'test':'data/test-00000-of-00001-2526ab833e27e0ee.parquet'}

def load_role_data(directory, download=False):
    """Prefer local official JSON; otherwise download the pinned, access-controlled data."""
    directory=Path(directory)
    raw={}
    for split in FILES:
        path=directory/f'{split}.json'
        if path.exists():
            raw[split]=json.loads(path.read_text(encoding='utf8'))
        elif download:
            import pyarrow.parquet as pq
            from huggingface_hub import hf_hub_download
            path=hf_hub_download('opennyaiorg/InRhetoricalRoles',FILES[split],repo_type='dataset',revision=REVISION)
            raw[split]=pq.read_table(path).to_pylist()
        else:
            raise FileNotFoundError(f'Provide official {split}.json in {directory}, or use download=True after setting HF_TOKEN.')
    return prepare_frames(raw)

def prepare_frames(raw):
    frames,audit={},{}
    for split,documents in raw.items():
        frames[split],audit[split]=json_to_df(documents,split)
    for split in ['train','dev']:
        if frames[split].empty or frames[split].label.isna().any():
            raise ValueError(f'{split} must be nonempty and fully labeled')
    for a,b in [('train','dev'),('train','test'),('dev','test')]:
        overlap=document_hashes(frames[a]) & document_hashes(frames[b])
        audit[f'{a}_{b}_duplicate_documents']=len(overlap)
        if overlap:raise ValueError(f'Document leakage between {a} and {b}')
    return frames,audit

def demo_role_data():
    """Invented sentences for a software demonstration, never a reported benchmark."""
    phrases=['The court examined the evidence carefully.', 'The petitioner argued that the notice was late.',
      'The respondent submitted that the payment was made.', 'The parties signed a contract.',
      'The issue is whether the notice was valid.', 'The document continues here.',
      'In the Supreme Court before the bench.', 'That precedent does not apply to these facts.',
      'We rely on the earlier decision.', 'The rule requires a fair opportunity of hearing.',
      'The High Court had dismissed the petition.', 'We allow the present appeal.',
      'Section 12 provides for the limitation period.']
    raw={}
    for split,n in [('train',8),('dev',3),('test',2)]:
        docs=[]
        for i in range(n):
            spans=[];offset=0
            for role,phrase in zip(LABELS,phrases):
                text=f'{phrase} Synthetic {split} example {i}.'
                spans.append({'value':{'start':offset,'end':offset+len(text),'text':text,'labels':[role]}})
                offset+=len(text)+1
            docs.append({'annotations':[{'ground_truth':True,'result':spans}]})
        raw[split]=docs
    return prepare_frames(raw)

def save_frames(frames,audit,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    for name,frame in frames.items():frame.to_json(directory/f'{name}.jsonl',orient='records',lines=True)
    (directory/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8')

def read_frames(directory):
    import pandas as pd
    return {s:pd.read_json(Path(directory)/f'{s}.jsonl',lines=True) for s in FILES}
