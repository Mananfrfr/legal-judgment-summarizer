"""Private review workspaces, bounded processing and immutable machine evidence."""
from __future__ import annotations
import hashlib,io,json,os,re,secrets,shutil,threading,time,zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from functools import lru_cache
from stage44_brief import summarize_file

MAX_FILE_BYTES=50_000_000
MAX_BATCH_BYTES=100_000_000
MAX_FILES=10
MAX_PENDING=3
RETENTION_SECONDS=3600

@lru_cache(maxsize=1)
def trained_model():
    from stage44_model import FullDocumentRanker
    import torch
    torch.set_num_threads(max(1,int(os.environ.get('LEGAL_CPU_THREADS','4'))))
    encoder=os.environ.get('LEGAL_ENCODER_DIR')
    bundle=os.environ.get('LEGAL_MODEL_BUNDLE')
    if not encoder or not bundle:raise ValueError('The trained model is not configured. Set the two model paths or explicitly select evidence-only mode.')
    return FullDocumentRanker(encoder,bundle)

class ReviewStore:
    def __init__(self,root,processor=None):
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.jobs={};self.lock=threading.RLock()
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='legal-review')
        self.processor=processor
        self.stop=threading.Event()
        self.sweeper=threading.Thread(target=self._sweep,daemon=True)
        self.sweeper.start()

    def create(self,owner):
        with self.lock:
            active=sum(j['status'] in {'uploading','queued','running'} for j in self.jobs.values())
            if active>=MAX_PENDING:raise ValueError('The processing queue is full. Please retry after a job finishes.')
            if sum(j['owner']==owner for j in self.jobs.values())>=3:raise ValueError('Clear your current workspace before starting more jobs.')
            key=secrets.token_hex(16);folder=self.root/key;folder.mkdir()
            job=dict(id=key,owner=owner,folder=folder,status='uploading',created=time.time(),updated=time.time(),
                     files=[],results=[],errors=[],reviews={},cancelled=False,completed=0,message='Receiving documents')
            self.jobs[key]=job
            return job

    def owned(self,key,owner):
        with self.lock:
            j=self.jobs.get(key)
            if not j or not secrets.compare_digest(j['owner'],owner):raise KeyError('Workspace not found or expired')
            return j

    def start(self,key,owner,budget,ocr,evidence_only):
        j=self.owned(key,owner)
        with self.lock:
            j.update(status='queued',message='Waiting for the model',budget=budget,ocr=ocr,evidence_only=evidence_only)
        self.pool.submit(self._run,key)

    def _run(self,key):
        with self.lock:
            j=self.jobs.get(key)
            if not j:return
            if j['cancelled']:self._delete(key);return
            j.update(status='running',message='Loading model' if not j['evidence_only'] else 'Extracting evidence')
        try:
            model=None if j['evidence_only'] or self.processor else trained_model()
            for index,item in enumerate(j['files']):
                if j['cancelled']:break
                j['message']=f'Processing {index+1} of {len(j["files"])}: {item["name"]}'
                try:
                    result=(self.processor(item['path'],j['budget'],j['ocr'],j['evidence_only']) if self.processor
                            else summarize_file(item['path'],j['budget'],model,ocr=j['ocr']))
                    # Storage names are opaque; show the original basename only.
                    result['source_file']=item['name']
                    result['markdown']=__import__('stage44_brief').render_markdown(result)
                    j['results'].append(dict(index=index,name=item['name'],result=result))
                except Exception as exc:
                    j['errors'].append(dict(index=index,name=item['name'],message=self._safe_error(exc)))
                j['completed']=index+1;j['updated']=time.time()
            j.update(status='ready' if j['results'] else 'failed',message='Ready for source review' if j['results'] else 'No summary was produced')
            self._persist(j)
        except Exception as exc:
            j['errors'].append(dict(message=self._safe_error(exc)))
            j.update(status='failed',message='Processing stopped')
        finally:
            j['updated']=time.time()
            if j['cancelled']:
                with self.lock:self._delete(key)

    @staticmethod
    def _safe_error(exc):
        if isinstance(exc,ValueError):return str(exc)
        if isinstance(exc,RuntimeError) and 'Evidence verification' in str(exc):return 'Source verification failed. The summary was withheld.'
        return f'{type(exc).__name__}: processing failed. Check the document and model configuration.'

    def snapshot(self,key,owner):
        j=self.owned(key,owner)
        return dict(id=j['id'],status=j['status'],message=j['message'],completed=j['completed'],total=len(j['files']),
                    documents=[dict(index=x['index'],name=x['name'],missing=x['result']['audit']['missing_sections'],
                                    flags=x['result']['audit']['flags']) for x in j['results']],errors=j['errors'],
                    retention_seconds=RETENTION_SECONDS)

    def document(self,key,owner,index):
        j=self.owned(key,owner)
        entry=next((x for x in j['results'] if x['index']==index),None)
        if not entry:raise KeyError('Document not ready')
        return j,entry

    def review(self,key,owner,index,sections,notes):
        j,entry=self.document(key,owner,index)
        if set(sections)!=set(entry['result']['sections']):raise ValueError('The review must contain all four sections.')
        if any(not isinstance(v,str) or len(v)>30000 for v in sections.values()) or len(notes)>10000:raise ValueError('Review text exceeds its size limit.')
        with self.lock:
            previous=j['reviews'].get(str(index),{}).get('revision',0)
            review=dict(revision=previous+1,saved_at=time.time(),sections=sections,notes=notes,
                        original_source_sha256=entry['result']['source_sha256'],
                        label='Reviewer draft — edited text is not automatically source-verified',
                        automatic_source_verification=False)
            j['reviews'][str(index)]=review
            self._persist(j)
        return review

    def _persist(self,j):
        with self.lock:
            file=j['folder']/'review.json'
            temp=j['folder']/'review.tmp'
            temp.write_text(json.dumps(j['reviews'],ensure_ascii=False,indent=2),encoding='utf8')
            temp.replace(file)

    def export(self,key,owner):
        j=self.owned(key,owner)
        if j['status'] not in {'ready','failed'}:raise ValueError('Wait for processing to finish before exporting.')
        out=io.BytesIO()
        with self.lock,zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
            z.writestr('processing_report.json',json.dumps(self.snapshot(key,owner),indent=2))
            for entry in j['results']:
                prefix=f'document_{entry["index"]+1:02d}/';r=entry['result']
                z.writestr(prefix+'machine_evidence.json',json.dumps(r,ensure_ascii=False,indent=2))
                z.writestr(prefix+'machine_summary.md',r['markdown'])
                review=j['reviews'].get(str(entry['index']))
                if review:
                    z.writestr(prefix+'reviewer_draft.json',json.dumps(review,ensure_ascii=False,indent=2))
                    draft='# Reviewer draft\n\nEdited text is not automatically source-verified.\n\n'
                    draft+='\n\n'.join('## '+k+'\n\n'+v for k,v in review['sections'].items())
                    z.writestr(prefix+'reviewer_draft.md',draft+'\n\n## Reviewer notes\n\n'+review['notes'])
        return out.getvalue()

    def clear(self,owner):
        with self.lock:
            for key,j in list(self.jobs.items()):
                if j['owner']==owner:
                    j['cancelled']=True
                    if j['status'] not in {'running','uploading'}:self._delete(key)

    def abort_upload(self,key,owner):
        self.owned(key,owner)
        with self.lock:self._delete(key)

    def _delete(self,key):
        # Resolve and check the final deletion target; never accept a client path.
        if not re.fullmatch(r'[a-f0-9]{32}',key):raise ValueError('Invalid workspace identifier')
        folder=(self.root/key).resolve()
        if folder.parent!=self.root:raise ValueError('Cleanup target outside data directory')
        if folder.exists():shutil.rmtree(folder)
        self.jobs.pop(key,None)

    def cleanup(self):
        with self.lock:
            for key,j in list(self.jobs.items()):
                if j['status'] in {'ready','failed'} and time.time()-j['updated']>RETENTION_SECONDS:self._delete(key)
            # Completed workspaces from a prior server process are also bounded.
            for folder in self.root.iterdir():
                if folder.is_dir() and re.fullmatch(r'[a-f0-9]{32}',folder.name) and folder.name not in self.jobs and time.time()-folder.stat().st_mtime>RETENTION_SECONDS:
                    self._delete(folder.name)

    def _sweep(self):
        while not self.stop.wait(60):self.cleanup()

    def close(self):
        self.stop.set();self.pool.shutdown(wait=True,cancel_futures=True)
