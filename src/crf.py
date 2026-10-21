"""Optional historical CRF comparison; the deployed head uses softmax."""
import torch
from torch import nn

class LinearChainCRF(nn.Module):
    """From-role to-role transition scores; variable length via boolean mask."""
    def __init__(self,num_tags):
        super().__init__()
        self.num_tags=num_tags
        self.start=nn.Parameter(torch.zeros(num_tags))
        self.end=nn.Parameter(torch.zeros(num_tags))
        self.transitions=nn.Parameter(torch.zeros(num_tags,num_tags))
    def log_partition(self,emissions,mask):
        assert mask[:,0].all()
        scores=self.start+emissions[:,0]
        for t in range(1,emissions.shape[1]):
            step=torch.logsumexp(scores.unsqueeze(2)+self.transitions.unsqueeze(0)
                                  +emissions[:,t].unsqueeze(1),dim=1)
            scores=torch.where(mask[:,t].unsqueeze(1),step,scores)
        return torch.logsumexp(scores+self.end,dim=1)
    def gold_score(self,emissions,tags,mask):
        s=self.start[tags[:,0]]+emissions[:,0].gather(1,tags[:,0:1]).squeeze(1)
        for t in range(1,emissions.shape[1]):
            step=(self.transitions[tags[:,t-1],tags[:,t]]+
                  emissions[:,t].gather(1,tags[:,t:t+1]).squeeze(1))
            s=s+step*mask[:,t]
        last=mask.long().sum(1)-1
        last_tags=tags.gather(1,last.unsqueeze(1)).squeeze(1)
        return s+self.end[last_tags]
    def forward(self,emissions,tags,mask):
        # Mean negative log likelihood per *real sentence*, not per document.
        assert mask.dtype==torch.bool and mask[:,0].all()
        return (self.log_partition(emissions,mask)-self.gold_score(emissions,tags,mask)).sum()/mask.sum()
    @torch.no_grad()
    def decode(self,emissions,mask):
        scores=self.start+emissions[:,0]
        history=[]
        for t in range(1,emissions.shape[1]):
            candidate=scores.unsqueeze(2)+self.transitions.unsqueeze(0)+emissions[:,t].unsqueeze(1)
            best,backptr=candidate.max(dim=1)
            history.append(backptr)
            scores=torch.where(mask[:,t].unsqueeze(1),best,scores)
        best_last=(scores+self.end).argmax(1)
        result=[]
        for b in range(emissions.shape[0]):
            n=int(mask[b].sum().item())
            label=int(best_last[b].item())
            path=[label]
            for t in range(n-2,-1,-1):
                label=int(history[t][b,label].item())
                path.append(label)
            result.append(path[::-1])
        return result
