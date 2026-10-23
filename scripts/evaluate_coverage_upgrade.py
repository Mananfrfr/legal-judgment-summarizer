"""Development and reserved-case evaluation of coverage changes (no weight training)."""
import argparse, hashlib, importlib.util, json, random, subprocess, sys, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dataset_index import index_official_dataset
from summary_evaluation import normalized_text_hash, canonical_hash, brief_text, lead_summary, mean_interval
from fast_rouge import overlap_scores
OUT=ROOT/'outputs/coverage_upgrade'
BASELINE_COMMIT='6071e3c64f92199edd972c81f408bf5c0c1b2664'
ASSETS=ROOT/'models' if (ROOT/'models/encoder/model.safetensors').exists() else ROOT.parent/'stage4_upgrade/models'
def write(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf8')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf8'))
def baseline_modules():
    for name,expected in load(OUT/'protocol.json')['baseline_hashes'].items():
        if digest(OUT/'baseline'/name)!=expected:raise ValueError('Baseline snapshot modified')
    originals={n:sys.modules.get(n) for n in ['stage44_source','stage44_attribution','stage44_brief']};loaded={}
    try:
        for name in originals:
            spec=importlib.util.spec_from_file_location(name,OUT/'baseline'/f'{name}.py')
            module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);loaded[name]=module
    finally:
        for name,module in originals.items():
            if module is None:sys.modules.pop(name,None)
            else:sys.modules[name]=module
    return loaded
def prepare(archives):
    if (OUT/'protocol.json').exists():raise ValueError('Protocol already exists')
    splits=load(ROOT/'data/splits.json'); seen=set(splits['train_ids']+splits['dev_ids']); manifests=[]
    for p in sorted(archives.glob('stage3*results.zip')):
        with zipfile.ZipFile(p) as z:
            for n in z.namelist():
                if n.endswith('run_manifest.json'):
                    m=json.loads(z.read(n)); ids=set()
                    for k,v in m.items():
                        if k.endswith('_ids') and isinstance(v,list):ids.update(x for x in v if isinstance(x,str))
                    seen.update(ids);manifests.append(dict(archive=p.name,sha256=digest(p),ids=sorted(ids)))
    if not manifests and (ROOT/'evaluation/coverage_upgrade_protocol.json').exists():
        manifests=load(ROOT/'evaluation/coverage_upgrade_protocol.json')['exposure_manifests']
        for m in manifests:seen.update(m['ids'])
    if len(manifests)<5:raise ValueError('Need Stage 3.2 through 3.6 exposure manifests')
    with zipfile.ZipFile(ROOT/'data/raw/dataset.zip') as z:
        pairs,_,_=index_official_dataset(z)
        hashes={r['id']:normalized_text_hash(z.read(r['judgment_path']).decode('utf-8-sig')) for r in pairs['train']+pairs['test']}
        forbidden={hashes[k] for k in seen if k in hashes}|{hashes[r['id']] for r in pairs['test']}
        available=[];unique=set()
        for r in pairs['train']:
            h=hashes[r['id']]
            if r['id'] not in seen and h not in forbidden and h not in unique:available.append(r);unique.add(h)
        holdout=random.Random(20261022).sample(available,100)
        dev=[r for r in pairs['train'] if r['id'] in splits['dev_ids']]
        for r in dev+holdout:
            r['source_sha256']=hashlib.sha256(z.read(r['judgment_path'])).hexdigest()
            r['reference_sha256']=hashlib.sha256(z.read(r['summary_path'])).hexdigest()
    baseline=OUT/'baseline';baseline.mkdir(parents=True,exist_ok=True)
    names=subprocess.check_output(['git','ls-tree','--name-only',BASELINE_COMMIT+':src'],cwd=ROOT,text=True).splitlines()
    for name in names:
        if name.endswith('.py'):(baseline/name).write_bytes(subprocess.check_output(['git','show',BASELINE_COMMIT+':src/'+name],cwd=ROOT))
    protocol=dict(seed=20261022,development=dev,reserved=holdout,exposure_manifests=manifests,
        baseline_commit=BASELINE_COMMIT,known_excluded_ids=len(seen),eligible_unique_pairs=len(available),baseline_hashes={p.name:digest(p) for p in baseline.glob('*.py')},
        limitations=['Same-corpus reserved sample, not external or human-validated accuracy.',
        'Exact normalized text duplicates excluded; near duplicates and encoder pretraining overlap not cleared.',
        'Official 100 test documents were inspected previously and are not used for tuning or this reserved evaluation.'])
    write(OUT/'protocol.json',protocol);print('Prepared',len(dev),'development and',len(holdout),'reserved cases; excluded',len(seen),'known IDs')
def generate(split):
    import torch
    from stage44_source import read_document
    from stage44_model import FullDocumentRanker
    p=load(OUT/'protocol.json');torch.set_num_threads(4);torch.manual_seed(42);old=baseline_modules()
    if split=='reserved':validate_freeze()
    model=FullDocumentRanker(ASSETS/'encoder',ASSETS/'stage4_english_model_bundle.zip',device='cuda',batch_size=3)
    write(OUT/split/'runtime.json',dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(0),encoder_signature=model.signature,batch_size=3))
    with zipfile.ZipFile(ROOT/'data/raw/dataset.zip') as z:
        for i,r in enumerate(p[split],1):
            dest=OUT/split/'cache'/r['id'].replace('.txt','.json')
            raw=z.read(r['judgment_path']);assert hashlib.sha256(raw).hexdigest()==r['source_sha256']
            path=OUT/split/'inputs'/r['id'];path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
            if not dest.exists():
                doc=old['stage44_source'].read_document(path,ocr='off');pred,info=model.score_document(doc)
                write(dest,dict(case_id=r['id'],source_sha256=r['source_sha256'],sentences=doc.sentences,predictions=pred,model=info))
            cache=load(dest);doc=read_document(path,ocr='off');candidate=OUT/split/'candidate_cache'/dest.name
            if [s['sentence'] for s in doc.sentences]!=[s['sentence'] for s in cache['sentences']]:
                if not candidate.exists() or [s['sentence'] for s in load(candidate)['sentences']]!=[s['sentence'] for s in doc.sentences]:
                    pred,info=model.score_document(doc)
                    write(candidate,dict(case_id=r['id'],source_sha256=r['source_sha256'],sentences=doc.sentences,predictions=pred,model=info))
            print(f'{split} {i}/{len(p[split])} {r["id"]} {len(doc.sentences)} sentences',flush=True)
def score(split,label):
    from stage44_source import read_document
    from stage44_brief import build_brief,audit_brief
    old=baseline_modules()
    p=load(OUT/'protocol.json'); hashes={f.name:digest(f) for f in (ROOT/'src').glob('*.py')}
    if split=='reserved':
        validate_freeze()
        if (OUT/split/label/'report.json').exists():raise ValueError('Reserved report already exists')
    rows=[]
    with zipfile.ZipFile(ROOT/'data/raw/dataset.zip') as z:
        for r in p[split]:
            cache=load(OUT/split/'cache'/r['id'].replace('.txt','.json'))
            path=OUT/split/'inputs'/r['id'];assert digest(path)==r['source_sha256']==cache['source_sha256']
            doc=read_document(path,ocr='off')
            old_doc=old['stage44_source'].read_document(path,ocr='off')
            candidate_cache=cache
            if [s['sentence'] for s in doc.sentences]!=[s['sentence'] for s in cache['sentences']]:candidate_cache=load(OUT/split/'candidate_cache'/r['id'].replace('.txt','.json'))
            outputs={}
            for budget in [800,400]:
                for method,fn,parsed,scored in [('baseline',old['stage44_brief'].build_brief,old_doc,cache),('candidate',build_brief,doc,candidate_cache)]:
                    if [s['sentence'] for s in parsed.sentences]!=[s['sentence'] for s in scored['sentences']]:raise ValueError('Sentence mismatch: independent model scores required')
                    predictions={int(k):v for k,v in scored['predictions'].items()}
                    brief=fn(parsed,budget,predictions,scored['model']);audit=audit_brief(parsed,brief)
                    outputs[f'{method}_{budget}']=dict(text=brief_text(brief),words=brief['summary_words'],empty_decision=not brief['sections']['decision'],empty_reasoning=not brief['sections']['reasoning'],source_pass=audit['exact_source_check_passed'],brief=brief)
                outputs[f'lead_{budget}']=dict(text=lead_summary(doc.sentences,budget))
            # Generate every candidate without access to its reference.
            write(OUT/split/label/'generated'/r['id'].replace('.txt','.json'),outputs)
        for r in p[split]:
            raw=z.read(r['summary_path']);assert hashlib.sha256(raw).hexdigest()==r['reference_sha256']
            outputs=load(OUT/split/label/'generated'/r['id'].replace('.txt','.json'))
            for method,o in outputs.items():
                rows.append(dict(case_id=r['id'],method=method,words=len(o['text'].split()),empty_decision=o.get('empty_decision'),empty_reasoning=o.get('empty_reasoning'),source_pass=o.get('source_pass'),**overlap_scores(raw.decode('utf-8-sig'),o['text'])))
    report=dict(split=split,cases=len(p[split]),hashes=hashes,methods={},paired={},
                baseline_commit=BASELINE_COMMIT,automatic_legal_accuracy=False,weights_retrained=False,
                scale='0 to 1; multiply by 100 for ROUGE points, not legal correctness percentages',
                limitations=p['limitations'],runtime=load(OUT/split/'runtime.json'))
    for method in sorted({r['method'] for r in rows}):
        rs=[r for r in rows if r['method']==method]
        report['methods'][method]={m:mean_interval([r[m] for r in rs]) for m in ['rouge1_f1','rouge2_f1','rougeL_f1']}
        report['methods'][method].update(mean_words=sum(r['words'] for r in rs)/len(rs),over_budget=sum(r['words']>int(method.split('_')[-1]) for r in rs),empty_decision=sum(r['empty_decision'] is True for r in rs),empty_reasoning=sum(r['empty_reasoning'] is True for r in rs),source_pass=sum(r['source_pass'] is True for r in rs))
    for budget in [800,400]:
        for comparator in ['baseline','lead']:
            a=[r['rougeL_f1'] for r in rows if r['method']==f'candidate_{budget}'];b=[r['rougeL_f1'] for r in rows if r['method']==f'{comparator}_{budget}']
            report['paired'][f'{budget}_minus_{comparator}']=mean_interval([x-y for x,y in zip(a,b)])
    write(OUT/split/label/'report.json',report);write(OUT/split/label/'per_case.json',rows)
    print(json.dumps({m:{k:(round(v['mean']*100,2) if isinstance(v,dict) else v) for k,v in d.items()} for m,d in report['methods'].items()},indent=2))
def freeze_state():
    assets=ASSETS
    return dict(hashes={f.name:digest(f) for f in (ROOT/'src').glob('*.py')},
                evaluator_sha256=digest(Path(__file__)),protocol_sha256=digest(OUT/'protocol.json'),
                bundle_sha256=digest(assets/'stage4_english_model_bundle.zip'),
                encoder_sha256={n:digest(assets/'encoder'/n) for n in ['config.json','tokenizer.json','model.safetensors']},
                primary='candidate_800 versus baseline_800 ROUGE-L; lead comparator; empty decisions and source checks',
                metric='rouge-score tokenization/Porter stemming; exact bit-parallel sequence LCS; 100 randomized equivalence checks',
                weights_retrained=False)
def validate_freeze():
    if load(OUT/'candidate_freeze.json')!=freeze_state():raise ValueError('Code, data protocol, evaluator or weights changed after freeze')
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','generate','score','freeze']);a.add_argument('--split',choices=['development','reserved'],default='development');a.add_argument('--label',default='v1');a.add_argument('--archives',type=Path,default=Path.home()/'Downloads');args=a.parse_args()
    if args.mode=='prepare':prepare(args.archives)
    elif args.mode=='generate':generate(args.split)
    elif args.mode=='score':score(args.split,args.label)
    else:
        if (OUT/'candidate_freeze.json').exists():raise ValueError('Already frozen')
        write(OUT/'candidate_freeze.json',freeze_state())
