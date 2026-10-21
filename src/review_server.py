"""Local-first HTTP review app. Remote binding requires explicit authentication."""
import base64,hmac,os,secrets,tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI,Request,UploadFile,File,Form,HTTPException
from fastapi.responses import HTMLResponse,Response,JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel,Field
from review_service import ReviewStore,MAX_FILE_BYTES,MAX_BATCH_BYTES,MAX_FILES

ROOT=Path(__file__).resolve().parent

class ReviewInput(BaseModel):
    sections:dict[str,str]
    notes:str=Field(default='',max_length=10000)

class BodyLimit:
    def __init__(self,app):self.app=app
    async def __call__(self,scope,receive,send):
        if scope['type']!='http':return await self.app(scope,receive,send)
        total=0
        async def limited_receive():
            nonlocal total
            message=await receive()
            if message['type']=='http.request':
                total+=len(message.get('body',b''))
                if total>MAX_BATCH_BYTES+1_000_000:raise HTTPException(413,'Upload exceeds the total request limit')
            return message
        await self.app(scope,limited_receive,send)

def create_app(data_root=None,processor=None,password=None):
    data_root=data_root or os.environ.get('LEGAL_REVIEW_DATA',str(Path(tempfile.gettempdir())/'legal_review_v45'))
    store=ReviewStore(data_root,processor)
    password=password if password is not None else os.environ.get('APP_PASSWORD','')
    @asynccontextmanager
    async def lifespan(app):
        store.cleanup()
        yield
        store.close()
    app=FastAPI(title='Judgment Review',docs_url=None,redoc_url=None,openapi_url=None,lifespan=lifespan)
    app.state.store=store
    app.add_middleware(BodyLimit)
    app.add_middleware(TrustedHostMiddleware,allowed_hosts=os.environ.get('APP_ALLOWED_HOSTS','localhost,127.0.0.1,testserver').split(','))

    @app.middleware('http')
    async def protect(request,call_next):
        if password:
            supplied=''
            try:
                kind,value=request.headers.get('authorization','').split(' ',1)
                if kind.lower()=='basic':supplied=base64.b64decode(value).decode().partition(':')[2]
            except (ValueError,UnicodeError):pass
            if not hmac.compare_digest(supplied.encode('utf8'),password.encode('utf8')):
                return Response(status_code=401,headers={'WWW-Authenticate':'Basic realm="Judgment Review"'})
        if request.method in {'POST','PUT','DELETE'}:
            if request.headers.get('x-review-request')!='1':return JSONResponse({'detail':'Missing request protection header'},403)
            origin=request.headers.get('origin')
            if origin and origin.rstrip('/')!=str(request.base_url).rstrip('/'):
                return JSONResponse({'detail':'Cross-origin writes are not allowed'},403)
            try:
                if int(request.headers.get('content-length','0'))>MAX_BATCH_BYTES+1_000_000:
                    return JSONResponse({'detail':'Upload exceeds the total request limit'},413)
            except ValueError:return JSONResponse({'detail':'Invalid content length'},400)
        owner=request.cookies.get('review_session','')
        if len(owner)!=64 or any(c not in '0123456789abcdef' for c in owner):owner=secrets.token_hex(32)
        request.state.owner=owner
        response=await call_next(request)
        response.set_cookie('review_session',owner,max_age=3600,httponly=True,samesite='strict',secure=request.url.scheme=='https')
        response.headers['Cache-Control']='no-store'
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Referrer-Policy']='no-referrer'
        response.headers['Content-Security-Policy']="default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; connect-src 'self'; frame-ancestors 'self'; object-src 'none'; base-uri 'none'"
        return response

    @app.exception_handler(KeyError)
    async def not_found(request,exc):return JSONResponse({'detail':'Workspace or document not found; it may have expired.'},404)
    @app.exception_handler(ValueError)
    async def bad_input(request,exc):return JSONResponse({'detail':str(exc)},400)

    @app.get('/',response_class=HTMLResponse)
    def index():return (ROOT/'review_app.html').read_text(encoding='utf8')

    @app.get('/api/health')
    def health():return dict(status='ok',version='4.5.0',model_configured=bool(os.environ.get('LEGAL_ENCODER_DIR') and os.environ.get('LEGAL_MODEL_BUNDLE')),public_accuracy_claim=False)

    @app.post('/api/jobs',status_code=202)
    async def upload(request:Request,files:list[UploadFile]=File(...),budget:int=Form(800),ocr:str=Form('auto'),evidence_only:bool=Form(False)):
        if not 1<=len(files)<=MAX_FILES:raise ValueError(f'Choose 1–{MAX_FILES} documents.')
        if not 100<=budget<=3000 or ocr not in {'auto','off','force'}:raise ValueError('Invalid summary settings')
        j=store.create(request.state.owner);total=0
        try:
            for index,file in enumerate(files):
                name=Path((file.filename or 'document').replace('\\','/')).name[:200]
                suffix=Path(name).suffix.lower()
                if suffix not in {'.pdf','.txt'}:raise ValueError('Only PDF and UTF-8 TXT documents are supported.')
                dest=j['folder']/f'input_{index:02d}{suffix}';size=0
                with dest.open('wb') as target:
                    while chunk:=await file.read(1024*1024):
                        size+=len(chunk);total+=len(chunk)
                        if size>MAX_FILE_BYTES or total>MAX_BATCH_BYTES:raise ValueError('Limit: 50 MB per document and 100 MB per batch.')
                        target.write(chunk)
                if not size:raise ValueError('An uploaded document is empty.')
                if suffix=='.pdf':
                    with dest.open('rb') as f:
                        if not f.read(1024).lstrip().startswith(b'%PDF-'):raise ValueError('A file named PDF does not contain a PDF header.')
                j['files'].append(dict(name=name,path=dest))
            store.start(j['id'],request.state.owner,budget,ocr,evidence_only)
            return store.snapshot(j['id'],request.state.owner)
        except Exception:
            store.abort_upload(j['id'],request.state.owner);raise
        finally:
            for file in files:await file.close()

    @app.get('/api/jobs/{key}')
    def status(key:str,request:Request):return store.snapshot(key,request.state.owner)

    @app.get('/api/jobs/{key}/documents/{index}')
    def document(key:str,index:int,request:Request):
        j,e=store.document(key,request.state.owner,index)
        return dict(result=e['result'],review=j['reviews'].get(str(index)),is_pdf=j['files'][index]['path'].suffix=='.pdf')

    @app.get('/api/jobs/{key}/documents/{index}/pages/{page}.png')
    def page_image(key:str,index:int,page:int,request:Request):
        j,e=store.document(key,request.state.owner,index)
        if not 1<=page<=len(e['result']['canonical_pages']):raise HTTPException(404,'Page not found')
        path=j['files'][index]['path']
        if path.suffix!='.pdf':raise HTTPException(404,'Text input has no PDF image')
        import fitz
        with fitz.open(path) as pdf:
            sheet=pdf[page-1]
            scale=min(1.5,(3_000_000/max(1,sheet.rect.width*sheet.rect.height))**.5)
            pix=sheet.get_pixmap(matrix=fitz.Matrix(scale,scale),alpha=False)
            return Response(pix.tobytes('png'),media_type='image/png')

    @app.put('/api/jobs/{key}/documents/{index}/review')
    def review(key:str,index:int,body:ReviewInput,request:Request):
        return store.review(key,request.state.owner,index,body.sections,body.notes)

    @app.get('/api/jobs/{key}/export')
    def export(key:str,request:Request):
        return Response(store.export(key,request.state.owner),media_type='application/zip',headers={'Content-Disposition':'attachment; filename="judgment_review.zip"'})

    @app.post('/api/clear')
    def clear(request:Request):
        store.clear(request.state.owner)
        return dict(cleared=True,message='Completed files removed. Active processing will stop after the current document and then remove its files.')
    return app

def main():
    import uvicorn
    host=os.environ.get('APP_HOST','127.0.0.1')
    if host not in {'127.0.0.1','localhost','::1'} and len(os.environ.get('APP_PASSWORD',''))<12:
        raise SystemExit('Remote binding requires APP_PASSWORD of at least 12 characters. Use HTTPS at your reverse proxy.')
    uvicorn.run(create_app(),host=host,port=int(os.environ.get('APP_PORT','7861')),access_log=False)

if __name__=='__main__':main()
