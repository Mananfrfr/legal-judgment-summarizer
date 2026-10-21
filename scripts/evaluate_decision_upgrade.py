"""Replay inspected development cases with verified cached model predictions.

These targeted source-text checks do not measure unseen-document accuracy.
Inputs and old predictions remain outside Git; only the compact report is public.
"""
from pathlib import Path
import argparse,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from stage44_source import read_document
from stage44_brief import build_brief,audit_brief,render_markdown

TARGETS={
    '129_2025':{
        'required':['They are accordingly dismissed','Tribunal must scrutinize','fifteen years','no order as to costs'],
        'excluded':[],
    },
    '37437_2026':{
        'required':['are, accordingly, set aside','Appellant shall be released forthwith, unless','appeal is, accordingly, allowed','within a month'],
        'excluded':['writ petition, however, came to be dismissed'],
    },
    '38471_2025':{
        'required':['appeal stands dismissed','interlocutory application'],
        'excluded':[],
    },
}

def checks(result,target):
    text=' '.join(s['sentence'] for s in result['sections']['decision']).casefold()
    return {'required':{value:value.casefold() in text for value in target['required']},
            'excluded':{value:value.casefold() not in text for value in target['excluded']}}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdf-dir',type=Path,required=True)
    parser.add_argument('--cache-dir',type=Path,required=True,help='Original predictions and baseline summaries')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    model_hash=hashlib.sha256((ROOT/'src/stage44_model.py').read_bytes()).hexdigest()
    signature=json.loads((ROOT/'models/manifest.json').read_text())['encoder_signature']
    rows=[]
    for prefix,target in TARGETS.items():
        files=list(args.pdf_dir.glob(prefix+'*.pdf'))
        if len(files)!=1:raise ValueError('Expected one source PDF for '+prefix)
        pdf=files[0];doc=read_document(pdf,ocr='off')
        baseline=json.loads((args.cache_dir/(pdf.stem+'.json')).read_text(encoding='utf8'))
        cache=json.loads((args.cache_dir/(pdf.stem+'.predictions.json')).read_text(encoding='utf8'))
        digest=hashlib.sha256(json.dumps([s['sentence'] for s in doc.sentences],ensure_ascii=False).encode()).hexdigest()
        key=hashlib.sha256((doc.sha256+digest+model_hash+signature).encode()).hexdigest()
        if cache['key']!=key or cache['sentence_digest']!=digest or baseline['source_sha256']!=doc.sha256:
            raise ValueError('Source/parser/model/cache mismatch for '+pdf.name)
        if cache['info']['encoder_signature']!=signature:raise ValueError('Encoder mismatch')
        predictions={int(k):v for k,v in cache['predictions'].items()}
        if set(predictions)!={s['id'] for s in doc.sentences}:raise ValueError('Incomplete cached predictions')
        result=build_brief(doc,800,predictions,cache['info'])
        result['audit']=audit_brief(doc,result)
        (args.output/(pdf.stem+'.json')).write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8')
        (args.output/(pdf.stem+'.md')).write_text(render_markdown(result),encoding='utf8')
        before,after=checks(baseline,target),checks(result,target)
        passed=lambda check:all(value for group in check.values() for value in group.values())
        rows.append({'source_file':pdf.name,'source_sha256':doc.sha256,'cached_predictions_verified':True,
                     'baseline_targeted_checks':before,'candidate_targeted_checks':after,
                     'baseline_all_targeted_checks_pass':passed(before),'candidate_all_targeted_checks_pass':passed(after),
                     'exact_source_check':result['audit']['exact_source_check_passed'],
                     'decision_source_pages':sorted({s['page'] for s in result['sections']['decision']}),
                     'flags':result['audit']['flags']})
        print(pdf.name,'baseline',passed(before),'candidate',passed(after),'source',result['audit']['exact_source_check_passed'],flush=True)
    report={'candidate':'4.6.0-preview','evaluation_kind':'inspected development regression; NOT independent or unseen evaluation',
            'weights_retrained':False,'predictions_reused_after_hash_validation':True,'legal_accuracy_measured':False,
            'checks_cover':'Selected final-decision wording and specified conditions only; not complete judgment quality',
            'remaining_work':'Independent attribution/completeness review and a new untouched evaluation set; Section 19 reasoning omission remains outside this fix',
            'documents':rows}
    (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    if not all(r['candidate_all_targeted_checks_pass'] and r['exact_source_check'] for r in rows):raise SystemExit(1)

if __name__=='__main__':main()
