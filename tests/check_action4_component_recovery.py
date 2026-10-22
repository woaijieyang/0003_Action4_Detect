import os
os.environ.update(CUDA_VISIBLE_DEVICES='GPU-8dfd78d1-6a27-4292-3338-9d33fffee9ef',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
import sys,json,copy
from pathlib import Path
R=Path('/data/zy/vln/a-test');O=R/'outputs/action4_independent_trials_v001';sys.path.insert(0,str(R/'src'))
from model.training.action4_component_train import *
from model.action4_component_experiments import aligned_iou_giou
from eval.action4_component_eval import evaluate

def same(a,b):
 if isinstance(a,torch.Tensor):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
 return a==b

def main():
 torch.set_num_threads(2);reports=[]
 for name in ('b0_giou_quality','a0_joint'):
  cfg=json.loads((R/f'configs/model/component_trials_v001/{name}.json').read_text());cfg['preprocess']=dict(cfg['preprocess'],full_size=[128,96],roi_size=64);cfg.update(micro=1,batch=1,loader_workers=0)
  seed(cfg['seed']);model=construct(cfg,'cuda:0');opt,sch=optimizer_for(model,cfg);rng=np.random.default_rng(777);data=ComponentData(cfg['validation_receipt'],'train',cfg['role']);items=[data[0]]
  first=update(model,opt,sch,items,cfg,'cuda:0',rng);path=O/(name+'_recovery_test.pt');save_checkpoint_atomic(path,state_for(model,opt,sch,cfg,rng,0,1,1,source_id()))
  second=update(model,opt,sch,[data[1]],cfg,'cuda:0',rng);expected={k:v.cpu().clone() for k,v in model.state_dict().items()};expected_opt=copy.deepcopy(opt.state_dict());expected_sch=copy.deepcopy(sch.state_dict())
  del model,opt,sch
  state=torch.load(path,map_location='cpu',weights_only=True);model=construct(cfg,'cuda:0');opt,sch=optimizer_for(model,cfg);rng=np.random.default_rng(0);position=restore(state,model,opt,sch,cfg,rng,source_id());del state
  repeated=update(model,opt,sch,[data[1]],cfg,'cuda:0',rng)
  assert position==(0,1,1) and second==repeated
  assert all(torch.equal(v,model.state_dict()[k].cpu()) for k,v in expected.items())
  assert same(expected_opt,opt.state_dict()) and same(expected_sch,sch.state_dict())
  bad=state_for(model,opt,sch,cfg,rng,0,1,1,source_id());bad['config_id']='wrong'
  try:restore(bad,model,opt,sch,cfg,rng,source_id());raise AssertionError('bad resume accepted')
  except ValueError:pass
  reports.append(dict(variant=name,losses=first['losses'],exact_next_update=True,optimizer_exact=True,scheduler_exact=True,position_exact=True,wrong_identity_rejected=True))
  del model,opt,sch,expected,expected_opt,bad
  torch.cuda.empty_cache()
 # GIoU must provide gradient for nonoverlap; identity gives zero.
 a=torch.tensor([[0.,0.,1.,1.]],requires_grad=True);b=torch.tensor([[2.,2.,3.,3.]])
 _,g=aligned_iou_giou(a,b);(1-g).sum().backward();assert a.grad.abs().sum()>0
 _,g=aligned_iou_giou(a.detach(),a.detach());assert torch.allclose(g,torch.ones_like(g))
 (O/'RECOVERY_TEST.json').write_text(json.dumps(dict(status='PASS',scope='real-data reduced-resolution engineering recovery; not accuracy',cases=reports,nonoverlap_giou_gradient=True),indent=2));print('PASS',reports,flush=True)

if __name__=='__main__':main()
