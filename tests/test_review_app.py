import io,json,sys,tempfile,time,unittest,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fastapi.testclient import TestClient
from review_server import create_app
from stage44_brief import summarize_file
from review_service import ReviewStore,RETENTION_SECONDS

TEXT=b'1. The issue before us is whether the limitation period applies.\n2. In our view the claim was filed within time.\n3. We allow the appeal and direct payment within six weeks.'
HEADERS={'x-review-request':'1'}

def process(path,budget,ocr,evidence):return summarize_file(path,budget,ocr='off')

class AppTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.app=create_app(self.tmp.name,process,password='')
        self.client=TestClient(self.app);self.client.__enter__()
    def tearDown(self):
        self.client.__exit__(None,None,None);self.tmp.cleanup()
    def upload(self,name='case.txt',data=TEXT):
        r=self.client.post('/api/jobs',files={'files':(name,data)},data={'evidence_only':'true','ocr':'off'},headers=HEADERS)
        self.assertEqual(r.status_code,202,r.text)
        key=r.json()['id']
        for _ in range(200):
            r=self.client.get('/api/jobs/'+key)
            if r.json()['status'] in {'ready','failed'}:return key,r.json()
            time.sleep(.02)
        self.fail('Processing timed out')

    def test_upload_review_and_export_preserve_machine_evidence(self):
        key,info=self.upload();self.assertEqual(info['status'],'ready')
        url=f'/api/jobs/{key}/documents/0'
        original=self.client.get(url).json()['result']
        self.assertTrue(original['audit']['exact_source_check_passed'])
        edited={k:'Reviewer correction: '+k for k in original['sections']}
        review=self.client.put(url+'/review',json={'sections':edited,'notes':'Check the deadline.'},headers=HEADERS)
        self.assertEqual(review.status_code,200,review.text)
        self.assertFalse(review.json()['automatic_source_verification'])
        after=self.client.get(url).json();self.assertEqual(after['result'],original)
        raw=self.client.get(f'/api/jobs/{key}/export')
        with zipfile.ZipFile(io.BytesIO(raw.content)) as z:
            self.assertIn('document_01/reviewer_draft.md',z.namelist())
            self.assertEqual(json.loads(z.read('document_01/machine_evidence.json')),original)
            self.assertFalse(any('input_' in p for p in z.namelist()))

    def test_another_session_cannot_read_or_export_job(self):
        key,_=self.upload()
        other=TestClient(self.app)
        try:
            for url in [f'/api/jobs/{key}',f'/api/jobs/{key}/documents/0',f'/api/jobs/{key}/export',f'/api/jobs/{key}/documents/0/pages/1.png']:
                self.assertEqual(other.get(url).status_code,404)
        finally:other.close()

    def test_clear_removes_files_and_results(self):
        key,_=self.upload()
        self.assertTrue((Path(self.tmp.name)/key).exists())
        self.assertEqual(self.client.post('/api/clear',headers=HEADERS).status_code,200)
        self.assertFalse((Path(self.tmp.name)/key).exists())
        self.assertEqual(self.client.get('/api/jobs/'+key).status_code,404)

    def test_rejects_wrong_file_and_cross_origin_mutation(self):
        for name,data in [('fake.pdf',b'not a PDF'),('script.html',b'<script>')]:
            r=self.client.post('/api/jobs',files={'files':(name,data)},headers=HEADERS)
            self.assertEqual(r.status_code,400)
        self.assertEqual(self.client.post('/api/clear').status_code,403)
        self.assertEqual(self.client.post('/api/clear',headers={**HEADERS,'origin':'https://another.example'}).status_code,403)
        self.assertFalse(list(Path(self.tmp.name).iterdir()))

    def test_missing_sections_are_visible_in_status(self):
        key,result=self.upload(data=b'1. The claimant seeks relief. The record is incomplete.')
        self.assertEqual(result['status'],'ready')
        self.assertIn('decision',result['documents'][0]['missing'])
        self.assertIn('MISSING_SECTION',result['documents'][0]['flags'])

    def test_paths_are_server_owned(self):
        key,_=self.upload(name='../../outside.txt')
        self.assertEqual(self.client.get(f'/api/jobs/{key}/documents/0').json()['result']['source_file'],'outside.txt')
        self.assertFalse((Path(self.tmp.name).parent/'outside.txt').exists())

    def test_expiry_does_not_remove_unrelated_directory(self):
        key,_=self.upload()
        unrelated=Path(self.tmp.name)/'keep';unrelated.mkdir()
        self.app.state.store.jobs[key]['updated']=time.time()-RETENTION_SECONDS-1
        self.app.state.store.cleanup()
        self.assertFalse((Path(self.tmp.name)/key).exists());self.assertTrue(unrelated.exists())

    def test_queue_is_bounded(self):
        store=self.app.state.store
        jobs=[store.create(str(i)) for i in range(3)]
        with self.assertRaises(ValueError):store.create('four')
        for i,j in enumerate(jobs):store.abort_upload(j['id'],str(i))

class PasswordTests(unittest.TestCase):
    def test_unicode_password_does_not_crash(self):
        import base64
        with tempfile.TemporaryDirectory() as tmp:
            with TestClient(create_app(tmp,process,password='passphrase-\u00e9-123')) as client:
                token=base64.b64encode('reviewer:passphrase-\u00e9-123'.encode()).decode()
                self.assertEqual(client.get('/',headers={'authorization':'Basic '+token}).status_code,200)
                self.assertEqual(client.get('/',headers={'authorization':'Basic ###'}).status_code,401)

    def test_password_gate_applies_to_health_and_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            with TestClient(create_app(tmp,process,password='test-passphrase-123')) as client:
                self.assertEqual(client.get('/').status_code,401)
                self.assertEqual(client.get('/api/health',auth=('reviewer','wrong')).status_code,401)
                self.assertEqual(client.get('/api/health',auth=('reviewer','test-passphrase-123')).status_code,200)

if __name__=='__main__':unittest.main()
