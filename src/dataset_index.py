"""Official IN-Abs / IN-Ext ZIP indexing; no archive extraction required."""

def dataset_tail(path, dataset_name):
    parts=path.replace('\\','/').split('/')
    aliases={dataset_name.lower(), dataset_name.lower().replace('-','')}
    hits=[i for i,p in enumerate(parts) if p.lower().replace('_','-') in aliases]
    return parts[hits[-1]+1:] if hits else None

def index_official_dataset(z):
    inabs={'train':{'judgement':{},'summary':{}},'test':{'judgement':{},'summary':{}}}
    inext={}
    for info in z.infolist():
        if info.is_dir() or not info.file_size or info.file_size>20_000_000: continue
        rel=dataset_tail(info.filename,'IN-Abs')
        if rel and len(rel)>=3:
            low=[x.lower() for x in rel]
            split='train' if low[0]=='train-data' else 'test' if low[0]=='test-data' else None
            typ=low[1]
            if split and typ in ('judgement','judgment','summary'):
                typ='judgement' if typ=='judgment' else typ
                key='/'.join(x.lower() for x in rel[2:])
                if key.endswith('.txt'):
                    inabs[split][typ][key]=info.filename
        rel=dataset_tail(info.filename,'IN-Ext')
        if rel and len(rel)>=2 and rel[0].lower() in ('judgement','judgment'):
            key='/'.join(x.lower() for x in rel[1:])
            if key.endswith('.txt'): inext[key]=info.filename
    pairs={}
    for split in ('train','test'):
        d=inabs[split]
        keys=sorted(set(d['judgement']) & set(d['summary']))
        pairs[split]=[{'id':key,'judgment_path':d['judgement'][key],
                       'summary_path':d['summary'][key], 'split':split} for key in keys]
    return pairs, inext, inabs
