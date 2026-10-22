"""Full-resolution next-batch parity from live-run recovery snapshots, no trainer interruption."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='GPU-8dfd78d1-6a27-4292-3338-9d33fffee9ef',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
from pathlib import Path
import sys,json,copy
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from model.training.action4_component_train import *

def equal(a,b):
 if isinstance(a,torch.Tensor):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b

def main():
 torch.set_num_threads(2);O=R/'outputs/action4_independent_trials_v001';results=[]
 for name in ('b0_giou_quality','a0_joint'):
  state=torch.load(O/name/'last.pt',map_location='cpu',weights_only=True);cfg=state['config'];epoch,cursor,step=state['epoch'],state['cursor'],state['step'];frozen=O/(name+'_live_recovery_input.pt');save_checkpoint_atomic(frozen,state);del state
  data=ComponentData(cfg['validation_receipt'],'train',cfg['role']);order=np.random.default_rng(cfg['seed']+epoch*1000003).permutation(len(data)).tolist();ids=order[cursor:cursor+cfg['batch']];items=[data[i] for i in ids];assert items
  expected=None
  for repeat in range(2):
   seed(cfg['seed']);model=construct(cfg,'cuda:0');opt,sch=optimizer_for(model,cfg);rng=np.random.default_rng(0);state=torch.load(frozen,map_location='cpu',weights_only=True);position=restore(state,model,opt,sch,cfg,rng,source_id());del state
   metrics=update(model,opt,sch,items,cfg,'cuda:0',rng)
   actual=dict(model={k:v.cpu().clone() for k,v in model.state_dict().items()},optimizer=copy.deepcopy(opt.state_dict()),scheduler=copy.deepcopy(sch.state_dict()),rng=rng_state(rng),metrics=metrics)
   if expected is None:expected=actual
   else:assert equal(expected,actual),'Full-size resume mismatch'
   del model,opt,sch;torch.cuda.empty_cache()
  results.append(dict(variant=name,parent_step=step,epoch=epoch,cursor=cursor,next_indices=ids,full_resolution=cfg['preprocess'],parameters_optimizer_scheduler_rng_and_losses_exact=True));del expected,actual,items,data
 (O/'LIVE_RECOVERY_TEST.json').write_text(json.dumps(dict(status='PASS',scope='same next actual full-size minibatches from live recovery state; training processes untouched',cases=results),indent=2));print('PASS',results)
if __name__=='__main__':main()
