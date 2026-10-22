"""Reproducible source-documented comparison; no accuracy claim or silent tuning."""
import argparse,hashlib,json,os,sys,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-dir',type=Path,default=ROOT/'src')
    parser.add_argument('--mode',choices=['prepare','score','replay'],required=True)
    parser.add_argument('--label',default='candidate')
    parser.add_argument('--encoder',type=Path)
    parser.add_argument('--bundle',type=Path)
    args=parser.parse_args()
    plan=json.loads(args.plan.read_text());out=args.output
    inputs=out/'inputs';inputs.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(args.source_dir.resolve()))
    if args.mode=='prepare':
        import fitz
        for item in plan['documents']:
            path=inputs/item['file']
            if not path.exists():
                request=urllib.request.Request(item['url'],headers={'User-Agent':'Mozilla/5.0'})
                with urllib.request.urlopen(request,timeout=90) as response:data=response.read()
                if not data.lstrip().startswith(b'%PDF-'):raise ValueError('Not a PDF: '+item['file'])
                path.write_bytes(data)
            with fitz.open(path) as pdf:
                print(item['file'],len(pdf),'pages',flush=True)
                (out/(path.stem+'_final_pages.txt')).write_text('\n'.join(p.get_text() for p in list(pdf)[-2:]),encoding='utf8')
                pdf[-1].get_pixmap(matrix=fitz.Matrix(1.25,1.25)).save(out/(path.stem+'_final_page.png'))
        freeze=dict(plan)
        freeze['documents']=[dict(item,sha256=hashlib.sha256((inputs/item['file']).read_bytes()).hexdigest()) for item in plan['documents']]
        freeze['candidate_module_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.source_dir.glob('stage44_*.py')}
        frozen=out/'pre_run_freeze.json'
        if frozen.exists():
            if json.loads(frozen.read_text())!=freeze:raise ValueError('Existing freeze differs; use a new output folder')
        else:frozen.write_text(json.dumps(freeze,indent=2),encoding='utf8')
        return
    from stage44_source import read_document
    from stage44_brief import build_brief,audit_brief,render_markdown
    model=None
    if args.mode=='score':
        import torch
        torch.set_num_threads(4)
        from stage44_model import FullDocumentRanker
        model=FullDocumentRanker(args.encoder,args.bundle)
    cache=out/'predictions';cache.mkdir(exist_ok=True)
    freeze=json.loads((out/'pre_run_freeze.json').read_text())
    model_hash=hashlib.sha256((args.source_dir/'stage44_model.py').read_bytes()).hexdigest()
    if model_hash!=freeze['candidate_module_hashes']['stage44_model.py']:raise ValueError('Model code differs from frozen scoring run')
    source_hashes={item['file']:item['sha256'] for item in freeze['documents']}
    signature=json.loads((ROOT/'models/manifest.json').read_text())['encoder_signature']
    results=out/args.label;results.mkdir(exist_ok=True)
    rows=[]
    for item in plan['documents']:
        start=time.monotonic();path=inputs/item['file'];doc=read_document(path,ocr='off')
        if doc.sha256!=source_hashes[path.name]:raise ValueError('Source PDF differs from frozen scoring run')
        digest=hashlib.sha256(json.dumps([s['sentence'] for s in doc.sentences],ensure_ascii=False).encode()).hexdigest()
        saved=cache/(path.stem+'.json')
        if model is not None:
            predictions,info=model.score_document(doc)
            data={'source_sha256':doc.sha256,'sentence_digest':digest,'predictions':predictions,'info':info}
            saved.write_text(json.dumps(data),encoding='utf8')
        else:
            data=json.loads(saved.read_text())
            if data['source_sha256']!=doc.sha256 or data['sentence_digest']!=digest:raise ValueError('Cache mismatch')
            predictions={int(k):v for k,v in data['predictions'].items()};info=data['info']
        if info['encoder_signature']!=signature:raise ValueError('Unexpected encoder fingerprint')
        if set(predictions)!={s['id'] for s in doc.sentences}:raise ValueError('Incomplete predictions')
        result=build_brief(doc,800,predictions,info);result['audit']=audit_brief(doc,result)
        (results/(path.stem+'.json')).write_text(json.dumps(result,ensure_ascii=False),encoding='utf8')
        (results/(path.stem+'.md')).write_text(render_markdown(result),encoding='utf8')
        rows.append({'file':path.name,'sha256':doc.sha256,'sentences':len(doc.sentences),'source_check':result['audit']['exact_source_check_passed'],
                     'decision_pages':sorted({e['page'] for e in result['sections']['decision']}),'flags':result['audit']['flags'],'seconds':round(time.monotonic()-start,1)})
        print(args.label,path.name,len(doc.sentences),'sentences',rows[-1]['seconds'],'seconds',flush=True)
    (results/'run_report.json').write_text(json.dumps({'label':args.label,'review':'source comparison still required','documents':rows},indent=2))

if __name__=='__main__':main()
