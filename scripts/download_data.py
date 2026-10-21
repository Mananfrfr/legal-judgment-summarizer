"""Download original datasets from their publishers; never mirror gated data."""
from pathlib import Path
import argparse,hashlib,sys,urllib.request
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',choices=['roles','inabs'])
    args=parser.parse_args()
    if args.dataset=='roles':
        from data_utils import load_role_data,save_frames
        frames,audit=load_role_data(ROOT/'data/raw/roles',download=True)
        save_frames(frames,audit,ROOT/'data/processed/official')
        print('Prepared the official role splits. HF_TOKEN is read from your environment or authorized HF login.')
    else:
        target=ROOT/'data/raw/dataset.zip';target.parent.mkdir(parents=True,exist_ok=True)
        expected='c77948fe580d26200f5c9fe53b81485d'
        if target.exists():
            with target.open('rb') as f:
                if hashlib.file_digest(f,'md5').hexdigest()==expected:print('Verified existing dataset.zip');return
            raise ValueError('Existing dataset.zip checksum differs; preserve it and resolve before retrying.')
        temporary=target.with_suffix('.partial')
        urllib.request.urlretrieve('https://zenodo.org/records/7152317/files/dataset.zip?download=1',temporary)
        with temporary.open('rb') as f:actual=hashlib.file_digest(f,'md5').hexdigest()
        if actual!=expected:raise ValueError('Downloaded dataset checksum mismatch')
        temporary.replace(target);print('Verified official IN-Abs/IN-Ext archive:',target)

if __name__=='__main__':main()
