"""Readable training entrypoints. New experiments never replace deployed checkpoints.

Architecture and principal hyperparameters follow the recorded experiments.
Consolidated training must be rerun before claiming numerical reproduction.
"""
from pathlib import Path
import json,math,random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score,f1_score,classification_report
from annotations import LABELS,LABEL2ID,ID2LABEL
from stage44_model import FullDocumentRanker,SentenceBiLSTM,checkpoint_signature

def seed_everything(seed=42):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def role_metrics(truth,prediction):
    return {'accuracy':float(accuracy_score(truth,prediction)),
            'macro_f1':float(f1_score(truth,prediction,labels=LABELS,average='macro',zero_division=0)),
            'weighted_f1':float(f1_score(truth,prediction,labels=LABELS,average='weighted',zero_division=0))}

def new_output(path):
    path=Path(path)
    if path.exists() and any(path.iterdir()):raise FileExistsError('Use a new output directory to preserve previous experiments.')
    path.mkdir(parents=True,exist_ok=True)
    return path

def train_baselines(train,dev,output):
    """Fit TF-IDF on TRAIN only, then compare four classifiers on DEV."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.dummy import DummyClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    import joblib
    output=new_output(output)
    vectorizer=TfidfVectorizer(max_features=5000)
    x_train=vectorizer.fit_transform(train.text)
    x_dev=vectorizer.transform(dev.text)
    models={'majority':DummyClassifier(strategy='most_frequent'),
            'knn':KNeighborsClassifier(n_neighbors=5),
            'logreg':LogisticRegression(max_iter=1000,class_weight='balanced',random_state=42),
            'svm':LinearSVC(max_iter=5000,class_weight='balanced',random_state=42)}
    rows=[]
    for name,model in models.items():
        model.fit(x_train,train.label)
        predictions=model.predict(x_dev)
        rows.append({'model':name,'split':'dev',**role_metrics(dev.label,predictions)})
        joblib.dump({'vectorizer':vectorizer,'model':model},output/f'{name}.joblib')
    result=pd.DataFrame(rows);result.to_csv(output/'metrics.csv',index=False)
    return result

class ContextInputs:
    """Same token packing as deployed inference; neighbors never cross a document."""
    _pack=FullDocumentRanker._pack
    def __init__(self,tokenizer):
        self.tokenizer=tokenizer
        self.mark_prev=tokenizer.encode('PREVIOUS:',add_special_tokens=False)
        self.mark_next=tokenizer.encode('NEXT:',add_special_tokens=False)
    def rows(self,frame,mode='context256'):
        if mode not in {'context256','target256'}:raise ValueError(mode)
        frame=frame.reset_index(drop=True)
        if not frame[['doc_index','sentence_index']].equals(frame[['doc_index','sentence_index']].sort_values(['doc_index','sentence_index']).reset_index(drop=True)):
            raise ValueError('Sort by document and sentence before encoding')
        tokens=self.tokenizer(frame.text.tolist(),add_special_tokens=False)['input_ids']
        rows=[]
        for i,target in enumerate(tokens):
            if mode=='target256':row=self.tokenizer(frame.text.iloc[i],truncation=True,max_length=256)
            else:
                previous=tokens[i-1] if i>0 and frame.doc_index.iloc[i-1]==frame.doc_index.iloc[i] else []
                following=tokens[i+1] if i+1<len(frame) and frame.doc_index.iloc[i+1]==frame.doc_index.iloc[i] else []
                row=self._pack(target,previous,following)
            row['labels']=LABEL2ID[frame.label.iloc[i]]
            rows.append(row)
        return rows

def train_encoder(train,dev,output,mode='context256',epochs=3,model_id='law-ai/InLegalBERT'):
    """Fine-tune the BERT sentence classifier; select the best epoch on DEV macro F1."""
    from transformers import AutoTokenizer,AutoModelForSequenceClassification,DataCollatorWithPadding,get_linear_schedule_with_warmup
    seed_everything();output=new_output(output)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer=AutoTokenizer.from_pretrained(model_id,use_fast=True)
    packer=ContextInputs(tokenizer)
    collator=DataCollatorWithPadding(tokenizer,return_tensors='pt')
    train_loader=DataLoader(packer.rows(train,mode),batch_size=8,shuffle=True,collate_fn=collator,generator=torch.Generator().manual_seed(42))
    dev_loader=DataLoader(packer.rows(dev,mode),batch_size=16,collate_fn=collator)
    model=AutoModelForSequenceClassification.from_pretrained(model_id,num_labels=13,id2label=ID2LABEL,label2id=LABEL2ID).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01)
    steps=math.ceil(len(train_loader)/4)*epochs
    scheduler=get_linear_schedule_with_warmup(optimizer,int(.1*steps),steps)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    history=[];best=-1
    for epoch in range(1,epochs+1):
        model.train();optimizer.zero_grad(set_to_none=True);losses=[]
        for step,batch in enumerate(train_loader):
            group_size=min(4,len(train_loader)-(step//4)*4)
            with torch.amp.autocast('cuda',enabled=device.type=='cuda'):
                loss=model(**{k:v.to(device) for k,v in batch.items()}).loss
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            scaler.scale(loss/group_size).backward();losses.append(loss.item())
            if (step+1)%4==0 or step+1==len(train_loader):
                scaler.unscale_(optimizer);nn.utils.clip_grad_norm_(model.parameters(),1.)
                before=scaler.get_scale();scaler.step(optimizer);scaler.update()
                if scaler.get_scale()>=before:scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        model.eval();pred=[]
        with torch.inference_mode():
            for batch in dev_loader:
                with torch.amp.autocast('cuda',enabled=device.type=='cuda'):
                    logits=model(**{k:v.to(device) for k,v in batch.items() if k!='labels'}).logits
                pred.extend(ID2LABEL[i] for i in logits.argmax(-1).tolist())
        metrics=role_metrics(dev.label,pred)
        history.append({'epoch':epoch,'loss':float(np.mean(losses)),**metrics});print(history[-1])
        if metrics['macro_f1']>best:
            best=metrics['macro_f1'];model.save_pretrained(output/'best_checkpoint');tokenizer.save_pretrained(output/'best_checkpoint')
            (output/'dev_metrics.json').write_text(json.dumps(metrics,indent=2))
    pd.DataFrame(history).to_csv(output/'history.csv',index=False)
    return output/'best_checkpoint'

@torch.inference_mode()
def document_embeddings(frame,encoder_directory):
    """Freeze the trained encoder and retain each judgment as its own sequence."""
    from transformers import AutoTokenizer,AutoModelForSequenceClassification,DataCollatorWithPadding
    tokenizer=AutoTokenizer.from_pretrained(encoder_directory,local_files_only=True)
    encoder=AutoModelForSequenceClassification.from_pretrained(encoder_directory,local_files_only=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');encoder.to(device).eval()
    packer=ContextInputs(tokenizer);collator=DataCollatorWithPadding(tokenizer,return_tensors='pt')
    sequences=[]
    for _,group in frame.groupby('doc_index',sort=False):
        rows=packer.rows(group.reset_index(drop=True));vectors=[]
        for start in range(0,len(rows),16):
            batch=collator([{k:v for k,v in r.items() if k!='labels'} for r in rows[start:start+16]])
            with torch.amp.autocast('cuda',enabled=device.type=='cuda'):
                hidden=encoder.base_model(**{k:v.to(device) for k,v in batch.items()}).last_hidden_state[:,0]
            vectors.append(hidden.float().cpu())
        sequences.append((torch.cat(vectors),torch.tensor([LABEL2ID[x] for x in group.label])))
    return sequences

def collate_documents(batch):
    lengths=torch.tensor([len(x) for x,y in batch])
    x=nn.utils.rnn.pad_sequence([x for x,y in batch],batch_first=True)
    y=nn.utils.rnn.pad_sequence([y for x,y in batch],batch_first=True,padding_value=0)
    mask=torch.arange(x.shape[1])[None,:]<lengths[:,None]
    return x,y,lengths,mask

def train_role_head(train_sequences,dev_sequences,output,encoder_directory,use_crf=False,epochs=25):
    """Compare independent softmax decisions with optional CRF sequence decoding."""
    from crf import LinearChainCRF
    seed_everything();output=new_output(output)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=SentenceBiLSTM().to(device);crf=LinearChainCRF(13).to(device) if use_crf else None
    parameters=list(model.parameters())+(list(crf.parameters()) if crf else [])
    optimizer=torch.optim.AdamW(parameters,lr=.001,weight_decay=.01)
    loader=DataLoader(train_sequences,batch_size=4,shuffle=True,collate_fn=collate_documents,generator=torch.Generator().manual_seed(42))
    dev_loader=DataLoader(dev_sequences,batch_size=4,collate_fn=collate_documents)
    best=-1;stale=0;history=[]
    for epoch in range(1,epochs+1):
        model.train()
        if crf:crf.train()
        for x,y,lengths,mask in loader:
            x,y,mask=x.to(device),y.to(device),mask.to(device)
            optimizer.zero_grad(set_to_none=True);logits=model(x,lengths)
            loss=crf(logits,y,mask) if crf else nn.functional.cross_entropy(logits[mask],y[mask])
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite head loss')
            loss.backward();nn.utils.clip_grad_norm_(parameters,1.);optimizer.step()
        model.eval();gold=[];pred=[]
        with torch.inference_mode():
            for x,y,lengths,mask in dev_loader:
                logits=model(x.to(device),lengths)
                paths=crf.decode(logits,mask.to(device)) if crf else [row[:n].tolist() for row,n in zip(logits.argmax(-1),lengths)]
                pred.extend(ID2LABEL[i] for path in paths for i in path)
                gold.extend(ID2LABEL[i] for i in y[mask].tolist())
        metrics=role_metrics(gold,pred);history.append({'epoch':epoch,**metrics});print(history[-1])
        if metrics['macro_f1']>best:
            best=metrics['macro_f1'];stale=0
            torch.save({'bilstm':model.state_dict(),'crf':crf.state_dict() if crf else None,
                        'feature_signature':checkpoint_signature(encoder_directory)},output/'best_head.pt')
            (output/'dev_metrics.json').write_text(json.dumps(metrics,indent=2))
        else:stale+=1
        if stale>=5:break
    pd.DataFrame(history).to_csv(output/'history.csv',index=False)
    return output/'best_head.pt'
