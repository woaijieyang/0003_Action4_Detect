"""Frozen-detector box-quality head;80full image epochs,20step snapshots."""
import json,math,time,os,fcntl,hashlib,traceback
import numpy as np
import torch
from data.action4_paths import PROJECT_ROOT as R
from data.action4_cached_box_quality import load_box_split
from model.action4_box_quality import BoxAwareQuality,varifocal_loss
from model.training.action4_runtime import atomic_json,save_checkpoint_atomic,canonical_hash

def publish(model,cfg,step,out):
    q=out/'eval';q.mkdir(exist_ok=True);request=q/f'step_{step:08d}.json'
    if request.exists():return
    while sum(p.name.count('.')==1 and not p.with_suffix('.done.json').exists() for p in q.glob('step_*.json'))>=8:
        if list(q.glob('*.failed.json')):raise RuntimeError('Eval failure')
        time.sleep(5)
    snapshot=out/f'eval_step_{step:08d}.pt';save_checkpoint_atomic(snapshot,{'schema':'action4.box_quality_eval.v1','head':model.state_dict(),'step':step,'config_id':canonical_hash(cfg),'resumable':False});atomic_json(request,{'step':step,'snapshot':str(snapshot.relative_to(R))})

def train(config,resume=None):
    cfg=json.loads((R/config).read_text());out=R/cfg['output'];guard=(out/'training.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for path,digest in cfg['source_identity'].items():
        if hashlib.sha256((R/path).read_bytes()).hexdigest()!=digest:raise ValueError('Source changed: '+path)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed']);data=load_box_split(R/cfg['cache_output'],'train');assert len(data)==1600
    model=BoxAwareQuality().to('cuda:0');opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=.0001);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,cfg['epochs']*math.ceil(len(data)/cfg['batch_images']),eta_min=cfg['lr']*.01);epoch=cursor=step=0
    if resume:
        state=torch.load(R/resume,map_location='cpu',weights_only=True)
        if state['schema']!='action4.box_quality_train.v1' or state['config_id']!=canonical_hash(cfg):raise ValueError('Resume identity mismatch')
        model.load_state_dict(state['head']);opt.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);epoch,cursor,step=state['epoch'],state['cursor'],state['step'];torch.set_rng_state(state['rng_cpu']);torch.cuda.set_rng_state_all(state['rng_cuda'])
    elif (out/'last.pt').exists():raise FileExistsError('Explicit resume required')
    atomic_json(out/'TRAIN_PROCESS.json',{'pid':os.getpid()});publish(model,cfg,step,out)
    for ep in range(epoch,cfg['epochs']):
        order=np.random.default_rng(cfg['seed']+ep*1000003).permutation(len(data));start=cursor if ep==epoch else 0
        for first in range(start,len(data),cfg['batch_images']):
            rows=[data[i] for i in order[first:first+cfg['batch_images']]];features=torch.cat([x['features'][x['supervision_valid']] for x in rows]).to('cuda:0');geometry=torch.cat([x['geometry'][x['supervision_valid']] for x in rows]).to('cuda:0');target=torch.cat([x['target'][x['supervision_valid']] for x in rows]).to('cuda:0')
            if not len(target):raise ValueError('Batch contains no supervised proposals')
            opt.zero_grad(set_to_none=True);loss=varifocal_loss(model(features,geometry),target)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite score loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();scheduler.step();step+=1;seen=first+len(rows);ne,nc=(ep+1,0) if seen==len(data) else (ep,seen)
            status={'step':step,'epoch':ep+1,'cursor_images':seen,'loss':float(loss.detach()),'gradient_norm':float(norm),'proposals':len(target),'positive_proposals':int((target>0).sum())};atomic_json(out/'STATUS.json',status)
            with (out/'train.jsonl').open('a') as f:f.write(json.dumps(status)+'\n')
            if step%cfg['eval_every']==0 or nc==0:save_checkpoint_atomic(out/'last.pt',{'schema':'action4.box_quality_train.v1','head':model.state_dict(),'config_id':canonical_hash(cfg),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'epoch':ne,'cursor':nc,'step':step,'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all(),'resumable':True})
            if step%cfg['eval_every']==0:publish(model,cfg,step,out)
        cursor=0
    publish(model,cfg,step,out);atomic_json(out/'TRAIN_COMPLETE.json',{'epochs':cfg['epochs'],'steps':step})
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--resume');a=p.parse_args()
    try:train(a.config,a.resume)
    except BaseException:
        cfg=json.loads((R/a.config).read_text());atomic_json(R/cfg['output']/'TRAIN_FAILURE.json',{'traceback':traceback.format_exc()});raise
