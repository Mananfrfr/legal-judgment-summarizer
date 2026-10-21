"""Original Stage 3.6 weak-label and budget rules, kept for reproducibility."""
import re
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
SUMMARY_WORD_BUDGET=400
SUMMARY_REPETITION_THRESHOLD=0.86
POSITIVE_FRACTION=0.16
SCORER=rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'],use_stemmer=True)
ROUGE_TOKENIZER=SCORER._tokenizer


def select_window_indices(n,limit):
    if n<=limit:return list(range(n))
    first=round(limit*0.5);last=round(limit*0.25);middle=limit-first-last
    mid_start=max(first,(n-middle)//2)
    idx=list(range(first))+list(range(mid_start,mid_start+middle))+list(range(n-last,n))
    assert len(idx)==len(set(idx))==limit
    return sorted(idx)

def unigram_bigram_counts(text):
    from collections import Counter
    toks=ROUGE_TOKENIZER.tokenize(text)
    return Counter(toks),Counter(zip(toks,toks[1:]))

def tfidf_weak_labels(sentences,reference,positive_frac=POSITIVE_FRACTION):
    """Stage 3.4's same positive-fraction sentence-ranking rule."""
    assert sentences
    tfidf=TfidfVectorizer(stop_words='english',ngram_range=(1,2),max_features=12000,
                          token_pattern=r'(?u)\b\w\w+\b')
    try:
        vectors=tfidf.fit_transform(list(sentences)+[reference])
        similarity=cosine_similarity(vectors[:-1],vectors[-1]).ravel()
    except ValueError:
        similarity=np.zeros(len(sentences),dtype=np.float32)
    eligible=[i for i,s in enumerate(sentences) if len(s.split())>=5]
    npos=min(len(eligible),max(1,round(len(eligible)*positive_frac)))
    rank=sorted(eligible,key=lambda i:(-similarity[i],i))
    y=np.zeros(len(sentences),dtype=np.float32)
    y[rank[:npos]]=1
    return y

def clipped_overlap_gain(used,candidate,reference_counts):
    """Improvement in clipped reference n-gram recall, counting repeats correctly."""
    return sum(min(count,max(0,reference_counts[gram]-used[gram]))
               for gram,count in candidate.items())

def greedy_oracle_labels(sentences,reference,budget=SUMMARY_WORD_BUDGET):
    """Training-only greedy ROUGE-1/2 *marginal coverage*, with word-budget and cost.
    This is NOT a ROUGE-L oracle and NOT a supervised human sentence annotation.
    It does not consume the dev labels unless called explicitly as a diagnostic.
    """
    from collections import Counter
    ref1,ref2=unigram_bigram_counts(reference)
    candidate_ngrams=[unigram_bigram_counts(s) for s in sentences]
    used1,used2=Counter(),Counter()
    selected=[];words_used=0;remaining=set(range(len(sentences)))
    while remaining:
        best_i=None;best_utility=0.0
        for i in sorted(remaining):
            wc=len(sentences[i].split())
            if wc<5 or wc>budget-words_used: continue
            c1,c2=candidate_ngrams[i]
            gain1=clipped_overlap_gain(used1,c1,ref1)/max(1,sum(ref1.values()))
            gain2=clipped_overlap_gain(used2,c2,ref2)/max(1,sum(ref2.values()))
            # Prioritize previously uncovered reference bigrams, with mild penalty for long sentences.
            utility=(0.4*gain1+0.6*gain2)/(wc**0.35)
            if utility>best_utility+1e-12:
                best_utility=utility;best_i=i
        if best_i is None: break  # No sentence adds relevant n-gram coverage within remaining budget.
        selected.append(best_i)
        words_used+=len(sentences[best_i].split())
        remaining.remove(best_i)
        c1,c2=candidate_ngrams[best_i]
        used1.update(c1);used2.update(c2)
    labels=np.zeros(len(sentences),dtype=np.float32)
    labels[selected]=1.0
    # If a human summary has no lexical match, don't silently create an all-negative training case.
    if not selected and sentences:
        fallback=tfidf_weak_labels(sentences,reference)
        if fallback.any():labels[int(np.flatnonzero(fallback)[0])]=1.0
    return labels

def tfidf_budgeted_labels(sentences,reference,budget=SUMMARY_WORD_BUDGET):
    """Budget-matched TF-IDF extraction teacher (training human summaries ONLY).
    Same 400-word constraint as greedy ROUGE; selects sentence importance labels.
    The reference is used to construct TRAIN labels, never as a model input.
    """
    assert sentences and budget>0
    try:
        vec=TfidfVectorizer(stop_words='english',ngram_range=(1,2),max_features=12000,
            token_pattern=r'(?u)\b\w\w+\b').fit_transform(list(sentences)+[reference])
        similarities=cosine_similarity(vec[:-1],vec[-1]).ravel()
    except ValueError:
        similarities=np.zeros(len(sentences),dtype=np.float32)
    ranking=sorted(range(len(sentences)),key=lambda i:(-float(similarities[i]),i))
    y=np.zeros(len(sentences),dtype=np.float32)
    used=0
    for i in ranking:
        wc=len(sentences[i].split())
        if wc<5 or wc>budget-used:continue
        y[i]=1.;used+=wc
        if used==budget:break
    return y

def sentence_tfidf_features(sentences):
    try:
        vec=TfidfVectorizer(stop_words='english',ngram_range=(1,2),max_features=8000).fit_transform(sentences)
        center=cosine_similarity(vec,np.asarray(vec.mean(axis=0)).ravel().reshape(1,-1)).ravel()
    except ValueError:
        vec=None;center=np.zeros(len(sentences),dtype=float)
    return vec,center

def budgeted_selection(sents,ranking,budget=SUMMARY_WORD_BUDGET,vectors=None):
    selected=[];words=0
    for idx in ranking:
        i=int(idx)
        wc=len(sents[i].split())
        if wc<5 or wc>budget or words+wc>budget:continue
        if vectors is not None and selected:
            if float(cosine_similarity(vectors[i],vectors[selected]).max())>=SUMMARY_REPETITION_THRESHOLD:
                continue
        selected.append(i);words+=wc
        if words>=budget:break
    return sorted(selected)

def lead_indices(sentences,budget=SUMMARY_WORD_BUDGET):
    selected=[];count=0
    for i,s in enumerate(sentences):
        n=len(s.split())
        if n<5:continue
        if count+n>budget:break  # Keep exact Stage 3.4 Lead behavior.
        selected.append(i);count+=n
    return selected

def position_features(case,with_roles):
    n=len(case['sentences'])
    pos=np.asarray(case['original_indices'],dtype=np.float32)[:,None]/max(1,case['original_sentence_count']-1)
    return np.concatenate([case['cls'].astype(np.float32),pos]+
        ([case['roles'].astype(np.float32)] if with_roles else []),axis=1)
