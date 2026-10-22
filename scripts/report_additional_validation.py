"""Compare frozen and revised decision coverage; these cases are development data."""
import argparse,hashlib,json,re
from pathlib import Path

def clean(text):
    return re.sub(r'\s+',' ',text).casefold()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    plan=json.loads((root/'evaluation/additional_validation_plan.json').read_text())
    targets=json.loads((root/'evaluation/additional_source_targets.json').read_text())
    freeze=json.loads((args.runs/'pre_run_freeze.json').read_text())
    current_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'src').glob('stage44_*.py')}
    assert current_hashes['stage44_model.py']==freeze['candidate_module_hashes']['stage44_model.py']
    rows=[]
    for item in plan['documents']:
        name=item['file'];stem=Path(name).stem;target=targets['targets'][name]
        row=dict(file=name,source_url=item['url'],source_target_pages=target['pages'],runs={})
        cache=json.loads((args.runs/'predictions'/(stem+'.json')).read_text())
        row.update(source_sha256=cache['source_sha256'],sentence_count=len(cache['predictions']),encoder_signature=cache['info']['encoder_signature'])
        for label in ['baseline','candidate_initial','revised_final']:
            result=json.loads((args.runs/label/(stem+'.json')).read_text(encoding='utf8'))
            assert result['source_sha256']==cache['source_sha256']
            decision=result['sections']['decision'];text=clean(' '.join(e['sentence'] for e in decision))
            checks={phrase:clean(phrase) in text for phrase in target['required']}
            pages=sorted({e['page'] for e in decision})
            row['runs'][label]=dict(required_text_checks=checks,all_required_text_present=all(checks.values()),decision_pages=pages,
                                    exact_source_check=result['audit']['exact_source_check_passed'],flags=result['audit']['flags'])
        rows.append(row)
    report=dict(date=plan['date'],version='4.6.0',scope='Specified operative-decision text coverage only; not accuracy or complete legal review',
                selection=plan['selection'],reviewer=targets['reviewer'],targets_timing=targets['timing'],
                status='Development cases: first-run failures motivated the revised pipeline; no untouched rerun claim',
                weights_retrained=False,actual_model_scoring=True,total_sentences_scored=sum(r['sentence_count'] for r in rows),
                comparisons_reuse_identical_predictions=True,
                documents_passing_specified_checks={label:sum(r['runs'][label]['all_required_text_present'] for r in rows) for label in ['baseline','candidate_initial','revised_final']},
                document_count=len(rows),frozen_candidate_commit=plan['frozen_candidate_commit'],baseline_commit=plan['baseline_commit'],
                frozen_module_hashes=freeze['candidate_module_hashes'],revised_module_hashes=current_hashes,
                remaining_failures=['APTEL issue section selects a question from a cited case; current controlling issue coverage remains unreliable.',
                                    'The earlier Section 19 reasoning omission is not fixed by this decision-only revision.',
                                    'No independent human full-summary evaluation or calibrated confidence; long sentences are truncated for model ranking.'],documents=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps({k:report[k] for k in ['documents_passing_specified_checks','document_count','total_sentences_scored']}))
    if not all(r['runs']['revised_final']['all_required_text_present'] and r['runs']['revised_final']['exact_source_check'] for r in rows):raise SystemExit(1)

if __name__=='__main__':main()
