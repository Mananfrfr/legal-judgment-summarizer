"""Freeze, generate, then score the current pipeline on all 100 IN-Abs test cases.

References are opened for scoring only after generation is complete. Do not tune
the pipeline on these scores. This reused public benchmark is not a blind test.
"""
import argparse,csv,hashlib,importlib.metadata,json,os,platform,subprocess,sys,time,zipfile
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dataset_index import index_official_dataset
from summary_evaluation import canonical_hash,normalized_text_hash,brief_text,lead_summary,overlap_scores,aggregate,mean_interval
CORE=['stage44_source.py','stage44_attribution.py','stage44_windows.py','stage44_model.py','stage44_brief.py']
METHODS=['pipeline_800','lead_800','pipeline_400','lead_400']

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf8')
    temporary.replace(path)

def archive_text(z,path):
    return z.read(path).decode('utf-8-sig',errors='strict')

def code_hashes():
    return {name:sha(ROOT/'src'/name) for name in CORE}

def prepare(args):
    target=args.output/'protocol.json'
    if target.exists():raise ValueError('Protocol already frozen; use generate/report or a separate run folder')
    with args.dataset.open('rb') as f:checksum=hashlib.file_digest(f,'md5').hexdigest()
    if checksum!='c77948fe580d26200f5c9fe53b81485d':raise ValueError('Official archive checksum mismatch')
    splits=json.loads((ROOT/'data/splits.json').read_text())
    seen_ids=set(splits['train_ids'])|set(splits['dev_ids'])
    with zipfile.ZipFile(args.dataset) as z:
        pairs,_,indexed=index_official_dataset(z)
        if len(pairs['test'])!=100:raise ValueError('Expected exactly 100 official test cases')
        train_hashes={};used_hashes={}
        train_sources=indexed['train']['judgement']
        if not seen_ids.issubset(train_sources):raise ValueError('Recorded train/dev IDs missing from archive')
        for case_id,path in train_sources.items():
            digest=normalized_text_hash(archive_text(z,path))
            train_hashes.setdefault(digest,[]).append(case_id)
            if case_id in seen_ids:used_hashes.setdefault(digest,[]).append(case_id)
        cases=[];test_hashes={};overlaps=[]
        for item in pairs['test']:
            text=archive_text(z,item['judgment_path']);norm=normalized_text_hash(text)
            if norm in test_hashes:raise ValueError('Duplicate normalized test documents')
            test_hashes[norm]=item['id']
            row=dict(item,source_sha256=hashlib.sha256(z.read(item['judgment_path'])).hexdigest(),
                     reference_sha256=hashlib.sha256(z.read(item['summary_path'])).hexdigest(),normalized_source_sha256=norm)
            cases.append(row)
            if norm in train_hashes:overlaps.append(dict(test_id=item['id'],train_ids=train_hashes[norm],used_for_selector=norm in used_hashes))
        if any(r['used_for_selector'] for r in overlaps):raise ValueError('Exact train/dev leakage detected; resolve before evaluation')
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),pipeline_version='4.6.0',
      inference_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
      dataset_url='https://zenodo.org/records/7152317',dataset_md5=checksum,source_code_sha256=code_hashes(),
      evaluator_sha256={p:sha(ROOT/p) for p in ['scripts/evaluate_current_summarizer.py','src/summary_evaluation.py']},
      model_bundle_sha256=sha(args.bundle),encoder_files_sha256={p:sha(args.encoder/p) for p in ['config.json','tokenizer.json','model.safetensors']},
      case_count=len(cases),cases=cases,methods=METHODS,budgets=[800,400],primary_method='pipeline_800',primary_metric='rougeL_f1',
      serialization='Displayed extract sentences in background, issue, reasoning, decision order; no captions, headings, warnings or omission suggestions',
      budget_policy='Unmodified app soft budget: whole sentences/operative conditions may overflow. Baseline uses consecutive parsed opening sentences, allowing one overlong first sentence.',
      scorer='rouge-score 0.1.2; default tokenizer; Porter stemming; case macro means; ROUGE-L sequence LCS, not ROUGE-Lsum',
      confidence='95% case-bootstrap percentile intervals, 10000 resamples, seed 42; paired bootstrap for pipeline-minus-lead differences',
      failure_policy='All 100 cases retained; generation failures score zero, reported separately; no silent exclusions',
      source_mode='Original publisher TXT input through the production parser; no PDF/OCR assessment',
      test_status='Reused official benchmark previously evaluated at Stage 3.6; current pipeline frozen before this run; not new blind or unseen-document accuracy',
      leakage=dict(archive_train_documents=len(train_sources),paired_train_documents=len(pairs['train']),recorded_selector_train_dev_ids=len(seen_ids),exact_normalized_test_train_overlaps=overlaps,
                   limits='No semantic near-duplicate or upstream encoder pretraining/corpus overlap clearance; exact text audit only'),
      automatic_legal_accuracy=False,weights_retrained=False)
    protocol['protocol_id']=canonical_hash(protocol)
    write(target,protocol)
    print('Frozen protocol:',protocol['protocol_id'],'cases:',len(cases),'exact overlaps:',len(overlaps),flush=True)

def validate(args):
    p=json.loads((args.output/'protocol.json').read_text(encoding='utf8'))
    original=p.pop('protocol_id');actual=canonical_hash(p);p['protocol_id']=original
    if original!=actual:raise ValueError('Frozen protocol was modified')
    if p['source_code_sha256']!=code_hashes():raise ValueError('Inference code changed after freeze')
    if any(sha(ROOT/name)!=digest for name,digest in p['evaluator_sha256'].items()):raise ValueError('Evaluator changed after freeze')
    return p

def generate(args):
    p=validate(args)
    if sha(args.bundle)!=p['model_bundle_sha256'] or any(sha(args.encoder/name)!=digest for name,digest in p['encoder_files_sha256'].items()):raise ValueError('Weights changed')
    import numpy as np,torch
    from stage44_model import FullDocumentRanker
    from stage44_source import read_document
    from stage44_brief import build_brief,audit_brief
    torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
    torch.backends.cudnn.benchmark=False
    model=FullDocumentRanker(args.encoder,args.bundle,device=args.device,batch_size=3)
    runtime=dict(protocol_id=p['protocol_id'],python=platform.python_version(),torch=torch.__version__,device=str(model.device),
                 cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0) if model.device.type=='cuda' else None,
                 transformers=importlib.metadata.version('transformers'),numpy=np.__version__,rouge_score=importlib.metadata.version('rouge-score'),
                 batch_size=model.batch_size,encoder_signature=model.signature,cpu_threads=4,
                 precision='Production CUDA autocast' if model.device.type=='cuda' else 'Production CPU float32',
                 deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    runtime_path=args.output/'runtime.json'
    if runtime_path.exists() and json.loads(runtime_path.read_text())!=runtime:raise ValueError('Cannot mix runtimes in one run')
    write(runtime_path,runtime)
    started=time.monotonic()
    with zipfile.ZipFile(args.dataset) as z:
        for index,item in enumerate(p['cases'],1):
            dest=args.output/'generated'/f'{index:03d}.json'
            if dest.exists():
                saved=json.loads(dest.read_text(encoding='utf8'))
                if saved['protocol_id']!=p['protocol_id'] or saved['case_id']!=item['id']:raise ValueError('Resume cache mismatch')
                continue
            stamp=time.monotonic();raw=z.read(item['judgment_path'])
            if hashlib.sha256(raw).hexdigest()!=item['source_sha256']:raise ValueError('Source differs from frozen data')
            # No reference summary is passed to parsing, scoring or selection.
            source=args.output/'inputs'/f'{index:03d}.txt';source.parent.mkdir(parents=True,exist_ok=True);source.write_bytes(raw)
            saved=dict(protocol_id=p['protocol_id'],case_id=item['id'],source_sha256=item['source_sha256'],status='ok',outputs={})
            try:
                doc=read_document(source,ocr='off')
                saved['sentence_count']=len(doc.sentences)
                for budget in p['budgets']:
                    saved['outputs']['lead_'+str(budget)]={'text':lead_summary(doc.sentences,budget),'status':'ok'}
                predictions,info=model.score_document(doc)
                if len(predictions)!=len(doc.sentences):raise ValueError('Incomplete model coverage')
                saved.update(sentence_count=len(doc.sentences),model=info,predictions=predictions)
                for budget in p['budgets']:
                    brief=build_brief(doc,budget,predictions,info);brief['audit']=audit_brief(doc,brief)
                    saved['outputs']['pipeline_'+str(budget)]={'text':brief_text(brief),'brief':brief,'status':'ok'}
            except Exception as exc:
                saved.update(status='failed',error=type(exc).__name__+': '+str(exc))
                if model.device.type=='cuda':torch.cuda.empty_cache()
            saved['seconds']=round(time.monotonic()-stamp,3);write(dest,saved)
            print(f'{index}/100 {item["id"]} {saved["status"]} sentences={saved.get("sentence_count",0)} seconds={saved["seconds"]} elapsed={time.monotonic()-started:.0f}s',flush=True)
    print('Generation complete; references have not been scored.',flush=True)

def report(args):
    p=validate(args)
    if any(not (args.output/'generated'/f'{i:03d}.json').exists() for i in range(1,101)):raise ValueError('Finish all 100 cases before scoring references')
    rows=[]
    with zipfile.ZipFile(args.dataset) as z:
        for index,item in enumerate(p['cases'],1):
            saved=json.loads((args.output/'generated'/f'{index:03d}.json').read_text(encoding='utf8'))
            if saved['protocol_id']!=p['protocol_id'] or saved['case_id']!=item['id'] or saved['source_sha256']!=item['source_sha256']:raise ValueError('Wrong generated record')
            raw=z.read(item['summary_path'])
            if hashlib.sha256(raw).hexdigest()!=item['reference_sha256']:raise ValueError('Reference changed')
            reference=raw.decode('utf-8-sig',errors='strict')
            if not reference.strip():raise ValueError('Empty official reference')
            for method in p['methods']:
                output=saved['outputs'].get(method,{});hypothesis=output.get('text','');brief=output.get('brief')
                row=dict(case_id=item['id'],method=method,status=output.get('status','failed'),budget=int(method.split('_')[-1]),
                         summary_words=len(hypothesis.split()),reference_words=len(reference.split()),
                         source_sentences=saved.get('sentence_count',0),
                         source_check=brief['audit']['exact_source_check_passed'] if brief else None,
                         missing_sections=';'.join(brief['audit']['missing_sections']) if brief else '',
                         flags=';'.join(brief['audit']['flags']) if brief else '',**overlap_scores(reference,hypothesis))
                rows.append(row)
    summary,lookup=aggregate(rows,[r['id'] for r in p['cases']],p['methods'])
    differences={}
    for budget in p['budgets']:
        differences[str(budget)]={metric:mean_interval([lookup[(c['id'],'pipeline_'+str(budget))][metric]-lookup[(c['id'],'lead_'+str(budget))][metric] for c in p['cases']])
                                  for metric in ['rouge1_f1','rouge2_f1','rougeL_f1']}
    result=dict(protocol_id=p['protocol_id'],case_count=100,complete=True,automatic_legal_accuracy=False,
                score_scale='0 to 1; multiply by 100 for ROUGE points, never interpret as percent legally correct',
                primary_method=p['primary_method'],primary_metric=p['primary_metric'],methods=summary,paired_pipeline_minus_lead=differences,
                audit={method:dict(exact_source_passes=sum(r['source_check'] is True for r in rows if r['method']==method),
                                   missing_section_cases=sum(bool(r['missing_sections']) for r in rows if r['method']==method)) for method in ['pipeline_800','pipeline_400']},
                runtime=json.loads((args.output/'runtime.json').read_text()),
                limitations=[p['test_status'],p['source_mode'],p['leakage']['limits'],'ROUGE measures lexical/reference overlap; it cannot establish factual, attribution or legal correctness.',
                             'Bootstrap intervals describe variation across these cases, not training exposure, domain shift, annotation uncertainty or numerical runtime equivalence.',
                             'App soft budgets can exceed requested length; reported length and precision/recall expose this tradeoff.'])
    write(args.output/'report.json',result)
    with (args.output/'per_case.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({m:{k:summary[m][k] for k in ['rouge1_f1','rouge2_f1','rougeL_f1','failures','mean_summary_words','over_budget_cases']} for m in p['methods']},indent=2))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','generate','report'])
    parser.add_argument('--dataset',type=Path,default=ROOT/'data/raw/dataset.zip')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/current_summary_benchmark')
    parser.add_argument('--encoder',type=Path,default=ROOT/'models/encoder')
    parser.add_argument('--bundle',type=Path,default=ROOT/'models/stage4_english_model_bundle.zip')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args()
    {'prepare':prepare,'generate':generate,'report':report}[args.mode](args)

if __name__=='__main__':main()
