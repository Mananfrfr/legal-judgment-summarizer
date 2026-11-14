"""Rebuild the faculty report from saved results; does not rerun inference."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ['rouge1', 'rouge2', 'rougeL']

def aggregate_rows(rows, expected_ids, methods):
    lookup = {(r['case_id'], r['method']): r for r in rows}
    expected = {(case, method) for case in expected_ids for method in methods}
    if len(lookup) != len(rows) or set(lookup) != expected:
        raise ValueError('Incomplete, duplicate or unexpected case/method records')
    result = {}
    for method in methods:
        result[method] = {}
        for metric in METRICS:
            for statistic in ['precision', 'recall', 'f1']:
                key = f'{metric}_{statistic}'
                values = [float(lookup[(case, method)][key]) for case in expected_ids]
                if not all(math.isfinite(x) and 0 <= x <= 1 for x in values):
                    raise ValueError('Invalid metric value: '+key)
                result[method][key] = sum(values) / len(values)
    return result

def main():
    paths = ['evaluation/coverage_upgrade_per_case.csv', 'evaluation/coverage_upgrade_reserved.json',
             'evaluation/coverage_upgrade_protocol.json', 'evaluation/role_results.csv']
    with (ROOT/paths[0]).open(newline='', encoding='utf8') as f:
        rows = list(csv.DictReader(f))
    saved = json.loads((ROOT/paths[1]).read_text())
    protocol = json.loads((ROOT/paths[2]).read_text())
    ids = [r['id'] for r in protocol['reserved']]
    if len(ids) != len(set(ids)) or len(ids) != saved['cases']:
        raise ValueError('Invalid case denominator')
    metrics = aggregate_rows(rows, ids, list(saved['methods']))
    for method, values in metrics.items():
        for metric in METRICS:
            if abs(values[metric+'_f1'] - saved['methods'][method][metric+'_f1']['mean']) > 1e-12:
                raise ValueError('Published F1 disagrees with per-case records')
    with (ROOT/paths[3]).open(newline='', encoding='utf8') as f:
        roles = list(csv.DictReader(f))
    report = ['# Faculty report: precision, recall and F1',
              'This report recomputes aggregates from saved evaluation records. It does not rerun the model or measure legal correctness.',
              '## 1. Summarization: 100 reserved IN-Abs judgments',
              'All metrics below are macro averages across the same 100 documents, shown on a 0–100 scale. ROUGE compares the summary with the dataset reference. F1 is computed per document, then averaged; it is not the harmonic mean of the two displayed averages.',
              '| Method | Metric | Precision | Recall | F1 |', '|---|---|---:|---:|---:|']
    for method in ['baseline_800', 'lead_800', 'candidate_800', 'baseline_400', 'lead_400', 'candidate_400']:
        for metric in METRICS:
            values = [metrics[method][metric+'_'+k]*100 for k in ['precision', 'recall', 'f1']]
            report.append(f'| {method} | {metric} | '+' | '.join(f'{v:.2f}' for v in values)+' |')
    report += ['`baseline` = previous 4.6 pipeline; `candidate` = upgraded 4.7 pipeline; `lead` = opening-sentence baseline. Suffixes are word targets.',
               '## 2. Sentence-role classification: recorded development results',
               '| Model | Accuracy (%) | Macro F1 (%) | Weighted F1 (%) |', '|---|---:|---:|---:|']
    for row in roles:
        report.append('| '+row['stage']+' / '+row['model']+' | '+' | '.join(f'{float(row[k])*100:.2f}' for k in ['accuracy','macro_f1','weighted_f1'])+' |')
    report += ['Role precision, recall, per-class support and the confusion matrix are not supplied in this aggregate CSV. They cannot be reconstructed from accuracy/F1 alone. Recompute them from saved ground-truth labels and predictions; do not invent them.',
               '## 3. Explain the metrics',
               '- Precision = TP / (TP + FP): how much of what was predicted is correct.',
               '- Recall = TP / (TP + FN): how much of the expected content was recovered.',
               '- F1 = 2 × precision × recall / (precision + recall): balances precision and recall.',
               '- For role classification, TP/FP/FN refer to class labels. Macro F1 gives each class equal weight; weighted F1 weights classes by their support.',
               '- For ROUGE-1/2, precision and recall count overlapping unigrams/bigrams. ROUGE-L uses longest-common-subsequence overlap. These are textual overlap metrics, not factual-correctness rates.',
               '## 4. Evidence and limitations',
               '- Candidate-800 ROUGE-L F1: 30.23 (95% case-bootstrap interval 28.57–31.96). Previous pipeline: 21.98; lead: 24.81, on these same cases.',
               '- 100/100 candidate summaries passed exact-source checks; 13/100 had an empty decision section. Neither 100% source matching nor 87% decision availability means summary accuracy.',
               '- The sample excludes known selector train/dev IDs and exact normalized duplicates. It remains same-corpus; upstream training overlap and semantic near duplicates are not cleared.',
               '- The initial freeze was amended for a synthetic quotation test before reserved scores were inspected. Both freezes and the amendment are preserved in evaluation/.',
               '- Existing weights were unchanged. The earlier official-test ROUGE-L 20.94 is from a different sample and must not be used as the before value for this table.',
               '- Outcome/amount/deadline rule checks are diagnostic screens, not a new human-validated factuality metric. No LLM judge score is included.',
               '## 5. What to say to faculty',
               '> Our role classifier recorded 65.41% macro F1 and 82.37% weighted F1 on development data. Our upgraded summarizer recorded 56.67 ROUGE-1 F1, 33.62 ROUGE-2 F1 and 30.23 ROUGE-L F1 on 100 reserved cases. We separately check source evidence and screen decisions, amounts and deadlines for possible errors. These checks do not establish a legal-correctness percentage.',
               '## Reproduction', 'Run `python scripts/build_faculty_metrics_report.py` from the repository. No model download, API key or third-party package is required.',
               'Inputs: '+', '.join('`'+p+'`' for p in paths)+'.']
    formatted=[]
    for line in report:
        grouped=formatted and ((line.startswith('|') and formatted[-1].startswith('|')) or
                               (line.startswith('- ') and formatted[-1].startswith('- ')))
        if formatted and not grouped:formatted.append('')
        formatted.append(line)
    (ROOT/'docs/faculty_metrics_report.md').write_text('\n'.join(formatted)+'\n', encoding='utf8')
    output = dict(case_count=len(ids),aggregation='Per-document mean of precision, recall and F1 independently',
                  automatic_legal_accuracy=False,methods=metrics,
                  source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths})
    (ROOT/'evaluation/faculty_metrics.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf8')
    print('Verified all case/method rows and saved F1 means. Wrote faculty report and metric provenance.')

if __name__ == '__main__': main()
