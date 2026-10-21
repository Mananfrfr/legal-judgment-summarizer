"""Execute default demos; optionally check one epoch of the synthetic word BiLSTM."""
from pathlib import Path
import argparse,json,os,sys
import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager
from jupyter_client import KernelManager
import tempfile
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--word-bilstm-smoke',action='store_true')
    args=parser.parse_args()
    os.environ['PROJECT_ROOT']=str(ROOT)
    os.environ['PROJECT_PYTHON']=sys.executable
    out=ROOT/'outputs/notebook_validation';out.mkdir(parents=True,exist_ok=True)
    rows=[]
    with tempfile.TemporaryDirectory() as tmp:
        kernel=Path(tmp)/'project-validation';kernel.mkdir()
        (kernel/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Project validation','language':'python'}))
        cases=[(file,False) for file in sorted((ROOT/'notebooks').glob('*.ipynb'))]
        if args.word_bilstm_smoke:
            cases.append((ROOT/'notebooks/00_classical_ml_and_word_bilstm.ipynb',True))
        for file,smoke in cases:
            nb=nbformat.read(file,as_version=4);nbformat.validate(nb)
            if smoke:
                for cell in nb.cells:
                    if cell.cell_type=='code':
                        cell.source=cell.source.replace('RUN_WORD_BILSTM = False','RUN_WORD_BILSTM = True').replace('WORD_EPOCHS = 15','WORD_EPOCHS = 1')
                nb.cells.append(nbformat.v4.new_code_cell('''%%project
from train_models import train_baselines
shared = train_baselines(train, dev, run / 'shared_baselines')
pd.testing.assert_frame_equal(current_scores, shared)
word_model.eval()
with torch.no_grad():
    short = word_model(torch.tensor([[2, 3, 0]], device=device), torch.tensor([2]))
    padded = word_model(torch.tensor([[2, 3, 0, 0, 0]], device=device), torch.tensor([2]))
    torch.testing.assert_close(short, padded)
unseen = make_dataset(pd.DataFrame({'text': ['never_seen_vocabulary_token_xyz', ''], 'label': [LABELS[0]] * 2}))
assert unseen.tensors[0][:, 0].tolist() == [1, 1]
assert unseen.tensors[1].tolist() == [1, 1]
assert len(history) == 1 and (run / 'word_bilstm.pt').exists()
print('PASS: baseline parity, one-epoch training, checkpoint reload, padding and unknown-word handling')
'''))
            km=KernelManager(kernel_name='project-validation',kernel_spec_manager=KernelSpecManager(kernel_dirs=[tmp]))
            client=NotebookClient(nb,km=km,timeout=300,resources={'metadata':{'path':str(ROOT)}})
            try:client.execute()
            finally:
                if km.has_kernel:km.shutdown_kernel(now=True)
            name=file.stem+'_word_smoke.ipynb' if smoke else file.name
            nbformat.write(nb,out/name)
            rows.append({'notebook':file.name,'default_demo_executed':not smoke,'synthetic_word_training_epochs':1 if smoke else 0,'heavy_training_executed':False})
            print('PASS',name,flush=True)
    (ROOT/'evaluation/notebook_validation.json').write_text(json.dumps(rows,indent=2))

if __name__=='__main__':main()
