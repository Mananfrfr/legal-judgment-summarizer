"""Fetch the project's frozen model release, verify SHA-256, and unpack safely."""
from pathlib import Path
import hashlib,json,urllib.request,zipfile
ROOT=Path(__file__).resolve().parents[1]

def main():
    manifest=json.loads((ROOT/'models/release_assets.json').read_text())
    destination=ROOT/'models';destination.mkdir(exist_ok=True)
    for asset in manifest['assets']:
        target=destination/asset['name']
        if not target.exists():
            temporary=target.with_suffix('.download')
            urllib.request.urlretrieve(asset['url'],temporary)
            with temporary.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
            if actual!=asset['sha256']:raise ValueError('Download hash mismatch for '+asset['name'])
            temporary.replace(target)
        with target.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        if actual!=asset['sha256']:raise ValueError('Existing model hash mismatch for '+asset['name'])
        if asset.get('extract'):
            with zipfile.ZipFile(target) as archive:
                for member in archive.infolist():
                    resolved=(destination/member.filename).resolve()
                    if not resolved.is_relative_to(destination.resolve()):raise ValueError('Unsafe archive path')
                archive.extractall(destination)
        print('Verified',asset['name'])

if __name__=='__main__':main()
