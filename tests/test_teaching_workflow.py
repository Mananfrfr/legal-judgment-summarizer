import itertools,json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from data_utils import demo_role_data,prepare_frames,save_frames,read_frames
from train_models import train_baselines,ContextInputs,collate_documents,train_role_head
from stage44_model import FullDocumentRanker
from selection_training import tfidf_budgeted_labels,greedy_oracle_labels,select_window_indices
from train_selector import train_selectors
from crf import LinearChainCRF
torch.set_num_threads(2)

class TeachingTests(unittest.TestCase):
    def test_demo_has_all_roles_and_separate_documents(self):
        frames,audit=demo_role_data()
        self.assertEqual(frames['train'].label.nunique(),13)
        self.assertEqual(audit['train_dev_duplicate_documents'],0)
        with tempfile.TemporaryDirectory() as tmp:
            save_frames(frames,audit,tmp)
            self.assertEqual(len(read_frames(tmp)['train']),104)

    def test_audit_rejects_duplicate_document_across_splits(self):
        record={'annotations':[{'result':[{'value':{'start':0,'end':4,'text':'Text','labels':['FAC']}}]}]}
        with self.assertRaises(ValueError):prepare_frames({s:[record] for s in ['train','dev','test']})

    def test_baseline_training_and_no_overwrite(self):
        frames,_=demo_role_data()
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'run'
            scores=train_baselines(frames['train'],frames['dev'],output)
            self.assertEqual(len(scores),4)
            self.assertTrue(scores.macro_f1.between(0,1).all())
            with self.assertRaises(FileExistsError):train_baselines(frames['train'],frames['dev'],output)

    def test_context_uses_deployed_packer(self):
        self.assertIs(ContextInputs._pack,FullDocumentRanker._pack)
        class Tokenizer:
            cls_token_id=101;sep_token_id=102
            def encode(self,text,**kwargs):return [11 if text.startswith('PREV') else 12]
            def __call__(self,texts,**kwargs):return {'input_ids':[[i+200]*5 for i,_ in enumerate(texts)]}
        frames,_=demo_role_data();rows=ContextInputs(Tokenizer()).rows(frames['train'])
        self.assertNotIn(11,rows[0]['input_ids'])
        self.assertNotIn(12,rows[12]['input_ids'])
        self.assertNotIn(11,rows[13]['input_ids'])

    def test_document_mask_ignores_padding(self):
        x,y,lengths,mask=collate_documents([(torch.zeros(2,768),torch.tensor([0,1])),(torch.zeros(1,768),torch.tensor([2]))])
        self.assertEqual(mask.tolist(),[[True,True],[True,False]])
        self.assertEqual(y[mask].tolist(),[0,1,2])

    def test_crf_matches_bruteforce_and_padding(self):
        crf=LinearChainCRF(3);em=torch.randn(1,3,3,requires_grad=True)
        mask=torch.tensor([[True,True,False]])
        scores=[crf.start[a]+em[0,0,a]+crf.transitions[a,b]+em[0,1,b]+crf.end[b] for a,b in itertools.product(range(3),repeat=2)]
        self.assertTrue(torch.allclose(crf.log_partition(em,mask)[0],torch.logsumexp(torch.stack(scores),0),atol=1e-5))
        expected=list(itertools.product(range(3),repeat=2))[torch.stack(scores).argmax()]
        self.assertEqual(crf.decode(em,mask)[0],list(expected))
        loss=crf(em,torch.tensor([[0,1,0]]),mask);loss.backward()
        self.assertTrue(torch.equal(em.grad[0,2],torch.zeros(3)))

    def test_weak_teachers_respect_budget(self):
        sentences=['The appeal is allowed with costs today.','The court requires payment within six weeks.','The claimant filed a claim before the tribunal.']
        for teacher in [tfidf_budgeted_labels,greedy_oracle_labels]:
            labels=teacher(sentences,'Payment is required within six weeks.',budget=14)
            self.assertGreater(labels.sum(),0)
            self.assertLessEqual(sum(len(s.split()) for s,y in zip(sentences,labels) if y),14)
        self.assertEqual(len(select_window_indices(350,120)),120)

    def test_both_role_heads_take_a_real_training_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);encoder=root/'encoder';encoder.mkdir()
            for n in ['config.json','tokenizer.json','model.safetensors']:(encoder/n).write_bytes(b'test fingerprint fixture')
            seq=[(torch.randn(3,768),torch.tensor([0,1,2])),(torch.randn(2,768),torch.tensor([1,2]))]
            for flag in [False,True]:
                output=train_role_head(seq,seq,root/str(flag),encoder,use_crf=flag,epochs=1)
                self.assertTrue(output.is_file())

    def test_all_four_selectors_train_on_fixture(self):
        sentences=['The claim was rejected by the tribunal.','The present appeal is allowed with costs.']
        case={'id':'synthetic','sentences':sentences,'original_indices':[0,1],'original_sentence_count':2,
              'cls':np.zeros((2,768),dtype=np.float32),'roles':np.ones((2,13),dtype=np.float32)/13,
              'reference':sentences[1],'tfidf_features':(None,None)}
        training={**case,'weak_labels':{n:np.array([0,1],dtype=np.float32) for n in ['tfidf_budgeted','greedy_rouge']}}
        with tempfile.TemporaryDirectory() as tmp:
            history=train_selectors({'train':[training],'dev':[case]},Path(tmp)/'run',epochs=1)
            self.assertEqual(history.model.nunique(),4)

if __name__=='__main__':unittest.main()
