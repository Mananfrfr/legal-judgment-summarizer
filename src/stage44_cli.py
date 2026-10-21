"""Batch entry point used by the isolated Colab environment and local CLI."""
import argparse
import csv
import json
from pathlib import Path
import zipfile
from stage44_brief import summarize_file

def safe_cell(value):
    value=str(value)
    return "'"+value if value.lstrip().startswith(('=','+','-','@')) else value

def run_batch(paths, output_dir, budget=800, model=None, model_choice=None, ocr='auto'):
    output=Path(output_dir)
    output.mkdir(parents=True,exist_ok=True)
    records,review=[],[]
    if len(paths)>30: raise ValueError('Maximum 30 input documents per batch.')
    for index,path in enumerate(paths,1):
        folder=output/f'judgment_{index:02d}'
        folder.mkdir(exist_ok=True)
        try:
            result=summarize_file(path,budget,model,model_choice,ocr)
            (folder/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
            (folder/'summary.md').write_text(result['markdown'],encoding='utf-8')
            records.append(dict(source_file=Path(path).name,processed=True,status=result['audit']['status'],
                                flags=result['audit']['flags'],words=result['summary_words'],
                                model=result['model'],result_folder=folder.name))
            for section,entries in result['sections'].items():
                review.append(dict(source_file=Path(path).name,section=section,
                                   evidence_ids=';'.join(str(e['id']) for e in entries),
                                   source_pages=';'.join(f'{e["page"]}-{e["page_end"]}' for e in entries),
                                   summary=' '.join(e['sentence'] for e in entries),
                                   machine_flags=';'.join(result['audit']['flags']),
                                   factuality_1_to_5='',completeness_1_to_5='',attribution_1_to_5='',
                                   critical_error_yes_no='',reviewer_notes=''))
        except Exception as exc:
            records.append(dict(source_file=Path(path).name,processed=False,status='failed',
                                error=f'{type(exc).__name__}: {exc}',result_folder=folder.name))
        print(f'[{index}/{len(paths)}] {Path(path).name}: {records[-1]["status"]}',flush=True)
    fields=['source_file','section','evidence_ids','source_pages','summary','machine_flags',
            'factuality_1_to_5','completeness_1_to_5','attribution_1_to_5','critical_error_yes_no','reviewer_notes']
    with (output/'manual_quality_review.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        writer.writerows({k:safe_cell(v) for k,v in row.items()} for row in review)
    report=dict(processed=sum(r['processed'] for r in records),failed=sum(not r['processed'] for r in records),
                machine_source_checks_passed=sum(r['status']=='source_checks_passed' for r in records),
                human_review_required=True,legal_accuracy_measured=False,documents=records)
    (output/'batch_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    archive=output.with_suffix('.zip')
    # Only this run's files: do not accidentally publish unrelated files in an output directory.
    files=[output/'batch_report.json',output/'manual_quality_review.csv']
    files += [output/r['result_folder']/name for r in records if r['processed'] for name in ['summary.json','summary.md']]
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for file in files: z.write(file,file.relative_to(output))
    return report,str(archive)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('inputs',nargs='+');p.add_argument('--output',required=True)
    p.add_argument('--encoder');p.add_argument('--bundle');p.add_argument('--model-choice')
    p.add_argument('--evidence-only',action='store_true')
    p.add_argument('--budget',type=int,default=800)
    p.add_argument('--ocr',choices=['auto','off','force'],default='auto')
    args=p.parse_args()
    model=None
    if not args.evidence_only:
        if not args.encoder or not args.bundle: p.error('Set --encoder and --bundle or explicitly choose --evidence-only.')
        from stage44_model import FullDocumentRanker
        model=FullDocumentRanker(args.encoder,args.bundle)
    paths=[]
    for value in args.inputs:
        path=Path(value)
        paths.extend(sorted(p for p in path.iterdir() if p.suffix.lower() in {'.pdf','.txt'}) if path.is_dir() else [path])
    if not paths: p.error('No PDF or TXT inputs found.')
    report,archive=run_batch(paths,args.output,args.budget,model,args.model_choice,args.ocr)
    print(json.dumps({k:v for k,v in report.items() if k!='documents'},indent=2))
    print('Results:',archive)
    if report['failed']: raise SystemExit(2)

if __name__=='__main__':main()
