"""Run notebook teaching cells in an isolated Python kernel to avoid Colab Hub conflicts.

Only the setup cell needs this machinery. %%project cells are ordinary Python,
share variables with one another, and forward their output to the notebook.
"""
from pathlib import Path
import atexit,hashlib,json,os,queue,subprocess,sys,tempfile,time

_runtime=None

def setup(root,training=True):
    global _runtime
    if _runtime is not None:return _runtime
    from IPython import get_ipython
    from IPython.display import display
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    root=Path(root).resolve()
    python=os.environ.get('PROJECT_PYTHON')
    if not python:
        env=root/'.notebook-env'
        python=str(env/('Scripts/python.exe' if os.name=='nt' else 'bin/python'))
        if not Path(python).exists():
            try:subprocess.check_call([sys.executable,'-m','venv',str(env)])
            except subprocess.CalledProcessError:
                subprocess.check_call([sys.executable,'-m','pip','install','-q','virtualenv'])
                subprocess.check_call([sys.executable,'-m','virtualenv',str(env)])
        requirement='requirements-training.txt'
        digest=hashlib.sha256((root/'requirements.txt').read_bytes()+(root/requirement).read_bytes()).hexdigest()
        stamp=env/'project-installed.json'
        if not stamp.exists() or json.loads(stamp.read_text()).get('digest')!=digest:
            subprocess.check_call([python,'-m','pip','install','-q','torch==2.8.0','--index-url','https://download.pytorch.org/whl/cu128' if Path('/content').exists() else 'https://download.pytorch.org/whl/cpu'])
            subprocess.check_call([python,'-m','pip','install','-q','-r',str(root/requirement)])
            subprocess.check_call([python,'-m','pip','check'])
            stamp.write_text(json.dumps({'digest':digest}))
    temporary=tempfile.TemporaryDirectory(prefix='legal-notebook-kernel-')
    directory=Path(temporary.name)/'legal-project';directory.mkdir()
    (directory/'kernel.json').write_text(json.dumps({'argv':[python,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Legal project (isolated)','language':'python'}))
    manager=KernelManager(kernel_name='legal-project',kernel_spec_manager=KernelSpecManager(kernel_dirs=[temporary.name]))
    env=dict(os.environ,PYTHONPATH=str(root/'src')+os.pathsep+os.environ.get('PYTHONPATH',''),PROJECT_ROOT=str(root),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    # Pass an authorized Colab secret only through the child environment, never output it.
    try:
        from google.colab import userdata
        if not env.get('HF_TOKEN'):env['HF_TOKEN']=userdata.get('HF_TOKEN')
    except Exception:pass
    manager.start_kernel(cwd=str(root),env=env)
    client=manager.client();client.start_channels();client.wait_for_ready(timeout=90)
    def execute(line,cell):
        message_id=client.execute(cell)
        error=None;deadline=time.monotonic()+8*60*60
        try:
            while time.monotonic()<deadline:
                try:msg=client.get_iopub_msg(timeout=1)
                except queue.Empty:continue
                if msg.get('parent_header',{}).get('msg_id')!=message_id:continue
                kind=msg['header']['msg_type'];content=msg['content']
                if kind=='stream':print(content['text'],end='')
                elif kind in {'display_data','execute_result'}:display(content['data'],raw=True)
                elif kind=='error':error='\n'.join(content['traceback'])
                elif kind=='status' and content['execution_state']=='idle':
                    if error:raise RuntimeError(error)
                    return
            raise TimeoutError('Training cell exceeded eight hours')
        except KeyboardInterrupt:
            manager.interrupt_kernel();raise
    def close():
        client.stop_channels();manager.shutdown_kernel(now=True);temporary.cleanup()
    get_ipython().register_magic_function(execute,'cell','project')
    atexit.register(close)
    _runtime=(manager,client,temporary)
    print('Isolated project Python is ready. Run the %%project cells in order.')
    return python
