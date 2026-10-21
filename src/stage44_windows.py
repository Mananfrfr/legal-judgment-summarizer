"""Bounded-context windows with auditable coverage; no 120-sentence subsampling."""
import numpy as np

def window_ranges(n, size=120, overlap=30):
    if n < 0 or size < 1 or not 0 <= overlap < size:
        raise ValueError('Invalid window dimensions.')
    start=0
    while start<n:
        end=min(n,start+size)
        yield start,end
        if end==n: break
        start=end-overlap

def aggregate_windows(n, evaluate, size=120, overlap=30):
    sums=None
    counts=np.zeros(n,dtype=np.int32)
    for a,b in window_ranges(n,size,overlap):
        values=np.asarray(evaluate(a,b),dtype=np.float32)
        if values.ndim!=2 or values.shape[0]!=b-a or not np.isfinite(values).all():
            raise ValueError('Invalid or non-finite model output; summary withheld.')
        if sums is None: sums=np.zeros((n,values.shape[1]),dtype=np.float32)
        if sums.shape[1]!=values.shape[1]: raise ValueError('Inconsistent model feature shape.')
        sums[a:b]+=values
        counts[a:b]+=1
    if n==0: return np.empty((0,0)),counts
    if not np.all(counts>0): raise RuntimeError('Some source sentences were not scored.')
    return sums/counts[:,None],counts
