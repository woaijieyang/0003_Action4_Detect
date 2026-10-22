"""Independent frozen-feature GDR-derived head training with atomic async eval."""
import json,math,time,os,fcntl,hashlib,traceback
import numpy as np
import torch
from data.action4_paths import PROJECT_ROOT as R
from data.action4_cached_surface_dataset import load_split
from model.action4_surface_head import SurfaceHead,surface_losses
from model.training.action4_runtime import atomic_json,save_checkpoint_atomic,canonical_hash


def publish(model,cfg,step,out):
    queue=out/'eval';queue.mkdir(exist_ok=True);request=queue/f'step_{step:08d}.json'
    if request.exists():return
    while sum(p.name.count('.')==1 and not p.with_suffix('.done.json').exists() for p in queue.glob('step_*.json'))>=8:
        if list(queue.glob('*.failed.json')):raise RuntimeError('Evaluation failure')
        time.sleep(5)
    path=out/f'eval_step_{step:08d}.pt'
    save_checkpoint_atomic(path,{'schema':'action4.surface_eval.v1','head':model.state_dict(),'step':step,'config_id':canonical_hash(cfg),'resumable':False})
    atomic_json(request,{'step':step,'snapshot':str(path.relative_to(R))})


def train(config,resume=None):
    cfg=json.loads((R/config).read_text());out=R/cfg['output'];out.mkdir(exist_ok=True)
    guard=(out/'training.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for path,digest in cfg['source_identity'].items():
        if hashlib.sha256((R/path).read_bytes()).hexdigest()!=digest:raise ValueError('Changed training source: '+path)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    features,targets,metadata=load_split(R/cfg['cache_output'],'train');assert len(features)==3024
    model=SurfaceHead().to('cuda:0');opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=.0001)
    total=cfg['epochs']*math.ceil(len(features)/cfg['batch']);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,total,eta_min=cfg['lr']*.01);epoch=cursor=step=0
    if resume:
        state=torch.load(R/resume,map_location='cpu',weights_only=True)
        if state.get('schema')!='action4.surface_train.v1' or state['config_id']!=canonical_hash(cfg):raise ValueError('Resume identity mismatch')
        model.load_state_dict(state['head']);opt.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);epoch,cursor,step=state['epoch'],state['cursor'],state['step'];torch.set_rng_state(state['rng_cpu']);torch.cuda.set_rng_state_all(state['rng_cuda'])
    elif (out/'last.pt').exists():raise FileExistsError('Explicit resume required')
    atomic_json(out/'TRAIN_PROCESS.json',{'pid':os.getpid(),'base_frozen':True});publish(model,cfg,step,out)
    for ep in range(epoch,cfg['epochs']):
        order=np.random.default_rng(cfg['seed']+ep*1000003).permutation(len(features));start=cursor if ep==epoch else 0
        for first in range(start,len(features),cfg['batch']):
            ids=order[first:first+cfg['batch']];opt.zero_grad(set_to_none=True);record={}
            for offset in range(0,len(ids),cfg['micro']):
                ix=ids[offset:offset+cfg['micro']];x=features[ix].to('cuda:0');y={k:v[ix].to('cuda:0') for k,v in targets.items()};losses=surface_losses(model(x),y);loss=sum(cfg['loss_weights'][k]*v for k,v in losses.items())
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite surface loss')
                weight=len(ix)/len(ids);(loss*weight).backward()
                for k,v in losses.items():record[k]=record.get(k,0)+float(v.detach())*weight
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();scheduler.step();step+=1;seen=first+len(ids);ne,nc=(ep+1,0) if seen==len(features) else (ep,seen)
            status={'step':step,'epoch':ep+1,'cursor':seen,'losses':record,'gradient_norm':float(norm)};atomic_json(out/'STATUS.json',status)
            with (out/'train.jsonl').open('a') as f:f.write(json.dumps(status)+'\n')
            if step%cfg['eval_every']==0 or nc==0:
                save_checkpoint_atomic(out/'last.pt',{'schema':'action4.surface_train.v1','head':model.state_dict(),'config_id':canonical_hash(cfg),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'epoch':ne,'cursor':nc,'step':step,'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all(),'resumable':True})
            if step%cfg['eval_every']==0:publish(model,cfg,step,out)
        cursor=0
    publish(model,cfg,step,out);atomic_json(out/'TRAIN_COMPLETE.json',{'epochs':cfg['epochs'],'steps':step})
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--resume');a=p.parse_args()
    try:train(a.config,a.resume)
    except BaseException:
        cfg=json.loads((R/a.config).read_text());atomic_json(R/cfg['output']/'TRAIN_FAILURE.json',{'traceback':traceback.format_exc()});raise
