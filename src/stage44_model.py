"""Frozen Stage 3.6 weights with full-document sliding-window inference.
Checkpoint hashes and strict state loading are retained. This is a new inference
regime; old held-out ROUGE scores do not automatically apply.
"""
from pathlib import Path
import hashlib, io, json, zipfile
import numpy as np
import torch
from torch import nn
from stage44_windows import window_ranges, aggregate_windows
LABELS = ['ANALYSIS','ARG_PETITIONER','ARG_RESPONDENT','FAC','ISSUE','NONE','PREAMBLE',
          'PRE_NOT_RELIED','PRE_RELIED','RATIO','RLC','RPC','STA']
MODEL_NAMES = {
    'Best ROUGE-1/2: TF-IDF trained (no roles)':'L120_tfidf_budgeted_no_roles',
    'Role-aware: Greedy ROUGE trained':'L120_greedy_rouge_roles',
}
ROLE_NAMES = {
    'FAC':'Facts','PREAMBLE':'Background','ISSUE':'Legal issues',
    'ARG_PETITIONER':'Petitioner arguments','ARG_RESPONDENT':'Respondent arguments',
    'ANALYSIS':'Court reasoning','RATIO':'Legal reasoning','RLC':'Lower court ruling',
    'RPC':'Present court ruling','STA':'Statutory provision','PRE_RELIED':'Precedent relied on',
    'PRE_NOT_RELIED':'Precedent not relied on','NONE':'Other',
}
SECTIONS = {'Facts / background': {'FAC','PREAMBLE'}, 'Legal issues': {'ISSUE'},
            'Arguments': {'ARG_PETITIONER','ARG_RESPONDENT'},
            'Reasoning / precedents': {'ANALYSIS','RATIO','PRE_RELIED','PRE_NOT_RELIED','STA'},
            'Rulings': {'RPC','RLC'}}


def checkpoint_signature(path: str | Path) -> str:
    """Exact fingerprint routine used for Stage 2.2 -> Stage 2.3/3.6 binding."""
    h=hashlib.sha256()
    for name in ('config.json','tokenizer.json','model.safetensors'):
        p=Path(path)/name
        if not p.is_file():
            raise FileNotFoundError(f'Missing trained encoder checkpoint file: {p}')
        h.update(name.encode());h.update(str(p.stat().st_size).encode())
        with p.open('rb') as f:
            h.update(f.read(128*1024))
            if p.stat().st_size>128*1024:
                f.seek(-128*1024,2)
                h.update(f.read(128*1024))
    return h.hexdigest()


def _digest(data:bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class SentenceBiLSTM(nn.Module):
    def __init__(self,in_dim=768,n_labels=13,hidden=128,dropout=0.3):
        super().__init__()
        self.norm=nn.LayerNorm(in_dim)
        self.drop=nn.Dropout(dropout)
        self.lstm=nn.LSTM(in_dim,hidden,num_layers=1,batch_first=True,bidirectional=True)
        self.classifier=nn.Linear(hidden*2,n_labels)
    def forward(self,x,lengths):
        x=self.drop(self.norm(x))
        packed=nn.utils.rnn.pack_padded_sequence(x,lengths.cpu(),batch_first=True,enforce_sorted=False)
        out,_=self.lstm(packed)
        padded,_=nn.utils.rnn.pad_packed_sequence(out,batch_first=True,total_length=x.shape[1])
        return self.classifier(self.drop(padded))


class SentenceSelector(nn.Module):
    def __init__(self,with_roles:bool):
        super().__init__()
        dim=768+1+(13 if with_roles else 0)
        self.net=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,160),nn.ReLU(),
                               nn.Dropout(0.25),nn.Linear(160,1))
    def forward(self,x):
        return self.net(x).squeeze(-1)


class FullDocumentRanker:
    """Load frozen checkpoints once, then run user PDFs/TXTs locally in Colab."""
    def __init__(self,encoder_dir,model_bundle,device=None,batch_size=12):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
        self.encoder_dir=Path(encoder_dir)
        self.model_bundle=Path(model_bundle)
        self.device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.batch_size=int(batch_size) if self.device.type=='cuda' else min(3,int(batch_size))
        self.signature=checkpoint_signature(self.encoder_dir)

        with zipfile.ZipFile(self.model_bundle) as z:
            manifest=json.loads(z.read('manifest.json'))
            if self.signature!=manifest['encoder_signature']:
                raise RuntimeError('Fine-tuned InLegalBERT fingerprint does not match the saved Stage 3.6 run. Do not use a base model or another checkpoint.')
            head_zip_bytes=z.read('stage2_3_bilstm_head_bundle.zip')
            if _digest(head_zip_bytes)!=manifest['head_zip_digest']:
                raise RuntimeError('BiLSTM head archive does not match the Stage 3.6 manifest.')
            model_data={k:z.read(k+'.pt') for k in MODEL_NAMES.values()}
            for k,blob in model_data.items():
                if _digest(blob)!=manifest['selector_sha256'][k]:
                    raise RuntimeError('Selector checkpoint hash mismatch for '+k)
        with zipfile.ZipFile(io.BytesIO(head_zip_bytes)) as head_zip:
            config=json.loads(head_zip.read('bilstm_softmax/head_config.json'))
            saved=torch.load(io.BytesIO(head_zip.read('bilstm_softmax/best_head.pt')),map_location='cpu',weights_only=True)
        if config['feature_checkpoint_signature']!=self.signature or saved['feature_signature']!=self.signature:
            raise RuntimeError('BiLSTM rhetorical-role head does not match encoder fingerprint.')
        self.head=SentenceBiLSTM(hidden=config['cfg']['hidden_size'],dropout=config['cfg']['dropout']).to(self.device)
        self.head.load_state_dict(saved['bilstm'],strict=True)
        self.head.eval()

        self.selectors={}
        for key,payload in model_data.items():
            ckpt=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=True)
            if ckpt['signature']!=manifest['experiment_signature'] or ckpt['experiment']!=key:
                raise RuntimeError('Selector experiment signature mismatch: '+key)
            if ckpt['max_sentences']!=120:
                raise RuntimeError('Stage 4 expects the original 120-sentence selector.')
            model=SentenceSelector(with_roles=bool(ckpt['use_roles'])).to(self.device)
            model.load_state_dict(ckpt['state_dict'],strict=True)
            self.selectors[key]=(model.eval(),bool(ckpt['use_roles']))

        self.tokenizer=AutoTokenizer.from_pretrained(str(self.encoder_dir),use_fast=True,local_files_only=True)
        self.mark_prev=self.tokenizer.encode('PREVIOUS:',add_special_tokens=False)
        self.mark_next=self.tokenizer.encode('NEXT:',add_special_tokens=False)
        self.collator=DataCollatorWithPadding(tokenizer=self.tokenizer,return_tensors='pt')
        self.encoder=AutoModelForSequenceClassification.from_pretrained(str(self.encoder_dir),local_files_only=True).to(self.device).eval()
        if self.encoder.config.num_labels!=13 or [self.encoder.config.id2label[i] for i in range(13)]!=LABELS:
            raise RuntimeError('Encoder config does not match our 13 OpenNyAI rhetorical roles.')
        for p in self.encoder.parameters():p.requires_grad_(False)

    def _pack(self,tgt,previous,nxt,max_length=256,target_cap=192):
        target=list(tgt[:min(target_cap,max_length-3)])
        budget=max_length-len(target)-3
        pmark=self.mark_prev if previous else []
        nmark=self.mark_next if nxt else []
        free=budget-len(pmark)-len(nmark)
        if free<0:raise ValueError('Invalid context token budget.')
        prev_cap=min(len(previous),(free+1)//2) if previous else 0
        next_cap=min(len(nxt),free//2) if nxt else 0
        left=free-prev_cap-next_cap
        prev_cap+=min(len(previous)-prev_cap,left)
        left=free-prev_cap-next_cap
        next_cap+=min(len(nxt)-next_cap,left)
        ctx=pmark+(list(previous[-prev_cap:]) if prev_cap else [])+nmark+(list(nxt[:next_cap]) if next_cap else [])
        ids=[self.tokenizer.cls_token_id]+target+[self.tokenizer.sep_token_id]+ctx+[self.tokenizer.sep_token_id]
        segment0=len(target)+2
        if len(ids)>256:raise RuntimeError('Token packing overflow.')
        return {'input_ids':ids,'attention_mask':[1]*len(ids),
                'token_type_ids':[0]*segment0+[1]*(len(ids)-segment0)}

    @torch.inference_mode()
    def _features(self,sentences:list[str],original_indices:list[int]):
        raw=self.tokenizer(sentences,add_special_tokens=False,padding=False,return_attention_mask=False)['input_ids']
        emb=np.empty((len(sentences),768),dtype=np.float32)
        for start in range(0,len(sentences),self.batch_size):
            end=min(start+self.batch_size,len(sentences))
            packed=[self._pack(raw[i],raw[i-1] if i and original_indices[i-1]+1==original_indices[i] else [],
                               raw[i+1] if i+1<len(sentences) and original_indices[i]+1==original_indices[i+1] else [])
                    for i in range(start,end)]
            batch={k:v.to(self.device) for k,v in self.collator(packed).items()}
            # Exactly the frozen Stage 3.6 feature extraction (CLS from base_model).
            with torch.amp.autocast(device_type='cuda',enabled=self.device.type=='cuda'):
                cls=self.encoder.base_model(**batch,return_dict=True).last_hidden_state[:,0,:]
            emb[start:end]=cls.float().cpu().numpy().astype(np.float16).astype(np.float32)
        starts=[0]+[i for i in range(1,len(sentences)) if original_indices[i]!=original_indices[i-1]+1]
        ends=starts[1:]+[len(sentences)]
        role_parts=[]
        for a,b in zip(starts,ends):
            logits=self.head(torch.from_numpy(emb[a:b].astype(np.float16).astype(np.float32)).unsqueeze(0).to(self.device),
                             torch.tensor([b-a],dtype=torch.long))[0]
            role_parts.append(logits.float().softmax(-1).cpu().numpy().astype(np.float16).astype(np.float32))
        return emb,np.concatenate(role_parts,axis=0)

    def score_document(self, document, model_choice=None):
        model_choice=model_choice or list(MODEL_NAMES)[0]
        if model_choice not in MODEL_NAMES:
            raise ValueError('Unknown selector checkpoint.')
        sentences=[s['sentence'] for s in document.sentences]
        n=len(sentences)
        if not n: raise ValueError('No source sentences to score.')
        selector,with_roles=self.selectors[MODEL_NAMES[model_choice]]
        def evaluate(a,b):
            cls,roles=self._features(sentences[a:b],list(range(a,b)))
            positions=np.arange(a,b,dtype=np.float32)[:,None]/max(1,n-1)
            features=np.concatenate([cls,positions]+([roles] if with_roles else []),axis=1)
            with torch.inference_mode():
                scores=selector(torch.from_numpy(features).to(self.device)).sigmoid().cpu().numpy()
            return np.concatenate([scores[:,None],roles],axis=1)
        values,counts=aggregate_windows(n,evaluate)
        predictions={}
        for i in range(n):
            r=int(values[i,1:].argmax())
            predictions[i]=dict(importance_score=float(values[i,0]),role_code=LABELS[r],
                                role_confidence=float(values[i,r+1]),windows_scored=int(counts[i]))
        lengths=[len(self.tokenizer.encode(s,add_special_tokens=False)) for s in sentences]
        long_ids=[i for i,k in enumerate(lengths) if k>192]
        info=dict(mode='trained_full_document',selector=MODEL_NAMES[model_choice],
                  encoder_signature=self.signature,source_sentences=n,scored_sentences=len(predictions),
                  sentence_coverage=1.0,window_size=120,overlap=30,
                  window_count=len(list(window_ranges(n))),long_sentence_ids=long_ids,
                  long_sentence_note='The encoder sees at most 192 target tokens per sentence, as trained. Full sentence text remains in the evidence engine.',
                  scores_are_calibrated=False,old_benchmark_scores_apply=False)
        return predictions,info

