"""Train the four Stage 3.6 selector conditions using fixed train/dev IDs.

Uses the existing frozen encoder and role head. Test references are never loaded
by this module. New checkpoints are research outputs, not automatic app updates.
"""
from pathlib import Path
import hashlib,json,zipfile
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from dataset_index import index_official_dataset
from selection_training import (select_window_indices,tfidf_budgeted_labels,greedy_oracle_labels,
    sentence_tfidf_features,budgeted_selection,position_features,SCORER)
from stage44_model import SentenceSelector
from train_models import new_output,seed_everything

def prepare_selector_cases(archive,split_file,ranker):
    """Read only fixed train/dev cases; use exact historical candidate sampling."""
    import spacy,re
    archive=Path(archive)
    with archive.open('rb') as f:digest=hashlib.file_digest(f,'md5').hexdigest()
    if digest!='c77948fe580d26200f5c9fe53b81485d':raise ValueError('Official dataset archive checksum mismatch')
    splits=json.loads(Path(split_file).read_text());nlp=spacy.blank('en');nlp.add_pipe('sentencizer')
    if set(splits['train_ids'])&set(splits['dev_ids']):raise ValueError('Train/dev IDs overlap')
    cases={'train':[],'dev':[]};seen={}
    with zipfile.ZipFile(archive) as z:
        pairs,_,_=index_official_dataset(z)
        lookup={x['id']:x for x in pairs['train']}
        test_ids={x['id'] for x in pairs['test']}
        if (set(splits['train_ids'])|set(splits['dev_ids']))&test_ids:raise ValueError('Official split IDs overlap')
        for split in cases:
            for key in splits[split+'_ids']:
                item=lookup[key]
                text=z.read(item['judgment_path']).decode('utf8',errors='replace').replace('\u00ad','')
                text=re.sub(r'(?<=\w)-\s*\n\s*(?=\w)','',text)
                text=re.sub(r'\s+',' ',text).strip()
                content_hash=hashlib.sha256(text.encode()).hexdigest()
                if content_hash in seen:raise ValueError('Duplicate judgment text: '+key)
                seen[content_hash]=split
                nlp.max_length=max(nlp.max_length,len(text)+100)
                sentences=[s.text.strip() for s in nlp(text).sents if re.search('[A-Za-z]',s.text)]
                indices=select_window_indices(len(sentences),120)
                chosen=[sentences[i] for i in indices]
                if not chosen:raise ValueError('No sentences: '+key)
                embeddings,roles=ranker._features(chosen,indices)
                case={'id':key,'sentences':chosen,'original_indices':indices,'original_sentence_count':len(sentences),
                      'cls':embeddings,'roles':roles,'reference':z.read(item['summary_path']).decode('utf8',errors='replace')}
                if split=='train':
                    case['weak_labels']={'tfidf_budgeted':tfidf_budgeted_labels(chosen,case['reference']),
                                         'greedy_rouge':greedy_oracle_labels(chosen,case['reference'])}
                case['tfidf_features']=sentence_tfidf_features(chosen)
                cases[split].append(case)
            print(split,len(cases[split]),'cases prepared')
    return cases

@torch.inference_mode()
def evaluate_selector(model,cases,with_roles,device):
    model.eval();scores=[]
    for case in cases:
        x=torch.from_numpy(position_features(case,with_roles)).to(device)
        importance=model(x).sigmoid().cpu().numpy()
        indices=budgeted_selection(case['sentences'],np.argsort(-importance,kind='stable'),400,case['tfidf_features'][0])
        summary='\n'.join(case['sentences'][i] for i in indices)
        scores.append(SCORER.score(case['reference'],summary)['rougeL'].fmeasure)
    return float(np.mean(scores))

def train_selectors(cases,output,epochs=16):
    """2 weak teachers × with/without predicted roles = 4 controlled conditions."""
    output=new_output(output);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if any('weak_labels' in c for c in cases['dev']):raise ValueError('DEV must not carry training labels')
    rows=[]
    for teacher in ['tfidf_budgeted','greedy_rouge']:
        for with_roles in [False,True]:
            seed_everything();name=f'L120_{teacher}_'+('roles' if with_roles else 'no_roles')
            model=SentenceSelector(with_roles).to(device)
            x=torch.from_numpy(np.concatenate([position_features(c,with_roles) for c in cases['train']]))
            y=torch.from_numpy(np.concatenate([c['weak_labels'][teacher] for c in cases['train']]))
            if not len(y) or not y.sum():raise ValueError('No positive weak labels')
            positive_weight=min(6.,max(1.,float((len(y)-y.sum())/y.sum())))
            loss_fn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([positive_weight],device=device))
            optimizer=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=.01)
            loader=DataLoader(TensorDataset(x,y),batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(42))
            best=-1;stale=0
            for epoch in range(1,epochs+1):
                model.train();losses=[]
                for xb,yb in loader:
                    optimizer.zero_grad(set_to_none=True)
                    loss=loss_fn(model(xb.to(device)),yb.to(device))
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite selector loss')
                    loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(loss.item())
                score=evaluate_selector(model,cases['dev'],with_roles,device)
                row={'model':name,'epoch':epoch,'train_loss':float(np.mean(losses)),'dev_rougeL':score}
                rows.append(row);print(row)
                if score>best+1e-5:
                    best=score;stale=0
                    torch.save({'state_dict':model.state_dict(),'experiment':name,'use_roles':with_roles,
                                'weak_label':teacher,'max_sentences':120,'val_rougeL':score,
                                'official_test_used':False,'research_only':True},output/(name+'.pt'))
                else:stale+=1
                if stale>=4:break
    result=pd.DataFrame(rows);result.to_csv(output/'history.csv',index=False)
    return result
