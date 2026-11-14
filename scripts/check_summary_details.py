"""Check a saved machine summary JSON against the original PDF/TXT, offline."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from stage44_source import read_document
from summary_checks import check_summary,checks_markdown

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--summary',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ocr',choices=['auto','off','force'],default='off')
    a=p.parse_args()
    doc=read_document(a.source,ocr=a.ocr)
    report=check_summary(doc,json.loads(a.summary.read_text(encoding='utf-8-sig')))
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'detail_checks.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8')
    (a.output/'detail_checks.md').write_text(checks_markdown(report),encoding='utf8')
    print(json.dumps({'status':report['status'],'flags':report['flags'],'legal_correctness_verified':False}))

if __name__=='__main__':main()
