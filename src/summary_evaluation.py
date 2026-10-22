"""Reference-overlap evaluation; these scores are not legal correctness rates."""
import hashlib
import json
import re
import numpy as np

SECTION_ORDER=('background','issue','reasoning','decision')

def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def normalized_text_hash(text):
    return hashlib.sha256(re.sub(r'\s+',' ',text).strip().casefold().encode()).hexdigest()

def brief_text(brief):
    """Only displayed extracts, in app order; no headings, audit or omitted passages."""
    return '\n'.join(e['sentence'] for section in SECTION_ORDER for e in brief['sections'][section])

def lead_summary(sentences,budget):
    """Keep the opening consecutive whole sentences; allow one overlong first sentence."""
    chosen=[];words=0
    for sentence in sentences:
        text=sentence['sentence'];length=len(text.split())
        if chosen and words+length>budget:break
        chosen.append(text);words+=length
    return '\n'.join(chosen)

def overlap_scores(reference,hypothesis):
    from rouge_score.rouge_scorer import RougeScorer
    scores=RougeScorer(['rouge1','rouge2','rougeL'],use_stemmer=True).score(reference,hypothesis)
    return {metric+'_'+name:float(getattr(score,attribute)) for metric,score in scores.items()
            for name,attribute in [('precision','precision'),('recall','recall'),('f1','fmeasure')]}

def mean_interval(values,seed=42,resamples=10000):
    """Case-level percentile bootstrap; it does not capture dataset/domain bias."""
    a=np.asarray(values,dtype=float)
    if a.ndim!=1 or not len(a) or not np.isfinite(a).all():raise ValueError('Need finite per-case values')
    rng=np.random.default_rng(seed)
    means=np.empty(resamples)
    for start in range(0,resamples,500):
        count=min(500,resamples-start)
        means[start:start+count]=a[rng.integers(0,len(a),size=(count,len(a)))].mean(axis=1)
    low,high=np.quantile(means,[.025,.975])
    return dict(mean=float(a.mean()),ci95_low=float(low),ci95_high=float(high),cases=len(a))

def aggregate(rows,expected_ids,methods,seed=42):
    """Reject incomplete/duplicate denominators; retain failed outputs as zero scores."""
    expected=set(expected_ids)
    if len(expected)!=len(expected_ids):raise ValueError('Duplicate expected IDs')
    lookup={}
    for row in rows:
        key=(row['case_id'],row['method'])
        if key in lookup:raise ValueError('Duplicate case/method result')
        lookup[key]=row
    if set(lookup)!={(case,method) for case in expected for method in methods}:
        raise ValueError('Missing or unexpected cases/methods')
    metrics=[m+'_'+s for m in ['rouge1','rouge2','rougeL'] for s in ['precision','recall','f1']]
    result={}
    for method in methods:
        selected=[lookup[(case,method)] for case in expected_ids]
        for row in selected:
            if row['status']!='ok' and any(row[k]!=0 for k in metrics):raise ValueError('Failed output must score zero')
        result[method]={k:mean_interval([r[k] for r in selected],seed) for k in metrics}
        result[method]['failures']=sum(r['status']!='ok' for r in selected)
        result[method]['mean_summary_words']=float(np.mean([r['summary_words'] for r in selected]))
        result[method]['over_budget_cases']=sum(r['summary_words']>r['budget'] for r in selected)
    return result,lookup
