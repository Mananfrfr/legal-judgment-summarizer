"""Execute all four default demo notebooks without training or dataset downloads."""
from pathlib import Path
import json,os,sys
import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager
from jupyter_client import KernelManager
import tempfile
ROOT=Path(__file__).resolve().parents[1]

def main():
    os.environ['PROJECT_ROOT']=str(ROOT)
    os.environ['PROJECT_PYTHON']=sys.executable
    out=ROOT/'outputs/notebook_validation';out.mkdir(parents=True,exist_ok=True)
    rows=[]
    with tempfile.TemporaryDirectory() as tmp:
        kernel=Path(tmp)/'project-validation';kernel.mkdir()
        (kernel/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Project validation','language':'python'}))
        for file in sorted((ROOT/'notebooks').glob('*.ipynb')):
            nb=nbformat.read(file,as_version=4);nbformat.validate(nb)
            km=KernelManager(kernel_name='project-validation',kernel_spec_manager=KernelSpecManager(kernel_dirs=[tmp]))
            client=NotebookClient(nb,km=km,timeout=300,resources={'metadata':{'path':str(ROOT)}})
            try:client.execute()
            finally:
                if km.has_kernel:km.shutdown_kernel(now=True)
            nbformat.write(nb,out/file.name)
            rows.append({'notebook':file.name,'default_demo_executed':True,'heavy_training_executed':False})
            print('PASS',file.name,flush=True)
    (ROOT/'evaluation/notebook_validation.json').write_text(json.dumps(rows,indent=2))

if __name__=='__main__':main()
