"""Exact sequence ROUGE-L using bit-parallel LCS for long legal documents.

Tokenization/stemming and ROUGE-1/2 remain rouge-score's defaults. This changes
computation cost, not the metric, and is checked against its reference DP.
"""
from rouge_score.rouge_scorer import RougeScorer

def lcs_length(a,b):
    masks={}
    for i,token in enumerate(a):masks[token]=masks.get(token,0)|(1<<i)
    state=0
    for token in b:
        matches=state|masks.get(token,0)
        state=matches & ~(matches-((state<<1)|1))
    return state.bit_count()

def overlap_scores(reference,hypothesis):
    scorer=RougeScorer(['rouge1','rouge2'],use_stemmer=True)
    result={metric+'_'+name:float(getattr(score,attr))
            for metric,score in scorer.score(reference,hypothesis).items()
            for name,attr in [('precision','precision'),('recall','recall'),('f1','fmeasure')]}
    a=scorer._tokenizer.tokenize(reference);b=scorer._tokenizer.tokenize(hypothesis)
    common=lcs_length(a,b);precision=common/len(b) if b else 0;recall=common/len(a) if a else 0
    result.update(rougeL_precision=precision,rougeL_recall=recall,
                  rougeL_f1=2*precision*recall/(precision+recall) if precision+recall else 0)
    return result
