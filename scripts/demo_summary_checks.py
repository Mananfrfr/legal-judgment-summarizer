"""Offline faculty demonstration with intentionally corrupted synthetic extracts."""
import json,sys
from copy import deepcopy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from stage44_source import read_document
from stage44_brief import summarize_file
from summary_checks import check_summary,checks_markdown

def main():
    out=ROOT/'outputs/correctness_demo';out.mkdir(parents=True,exist_ok=True)
    source=out/'invented_judgment.txt'
    source.write_text('1. The issue before us is whether the insurer must pay compensation.\n'
                      '2. We find that the policy covers this loss.\n'
                      '3. The appeal is allowed.\n'
                      '4. We direct the insurer to pay Rs. 2,00,000 within six weeks from today, subject to adjustment of earlier payments.\n',encoding='utf8')
    doc=read_document(source,ocr='off');brief=summarize_file(source,ocr='off')
    variants={'matching':brief}
    for label,before,after in [('wrong_outcome','appeal is allowed','appeal is dismissed'),
                                ('wrong_amount','2,00,000','9,00,000'),
                                ('wrong_deadline','six weeks','nine weeks'),
                                ('dropped_condition',', subject to adjustment of earlier payments','')]:
        changed=deepcopy(brief)
        for entry in changed['sections']['decision']:
            entry['sentence']=entry['sentence'].replace(before,after)
        variants[label]=changed
    omitted=deepcopy(brief)
    omitted['sections']['decision']=[e for e in omitted['sections']['decision'] if 'insurer to pay' not in e['sentence']]
    variants['omitted_direction']=omitted
    rows=[]
    for label,variant in variants.items():
        checks=check_summary(doc,variant)
        (out/(label+'.checks.json')).write_text(json.dumps(checks,indent=2,ensure_ascii=False),encoding='utf8')
        (out/(label+'.md')).write_text(checks_markdown(checks),encoding='utf8')
        rows.append(dict(example=label,status=checks['status'],flags=checks['flags']))
    (out/'matching.summary.json').write_text(json.dumps(brief,indent=2,ensure_ascii=False),encoding='utf8')
    (out/'demo_results.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
    print(json.dumps(rows,indent=2))
    assert rows[0]['status']=='no_rule_issue_detected'
    assert all('DECISION_SOURCE_MISMATCH' in row['flags'] for row in rows[1:5])
    assert 'POSSIBLE_DECISION_DETAIL_OMISSION' in rows[-1]['flags']
    print('PASS: synthetic checks only; no real-judgment correctness percentage measured.')

if __name__=='__main__':main()
