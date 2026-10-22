"""Frozen-best A0 auxiliary training; independent head checkpoints and eval queue."""
import json,os,time,math,fcntl,traceback,hashlib
from pathlib import Path
import numpy as np
import torch
from data.action4_paths import PROJECT_ROOT as R
from data.action4_component_data import ComponentData
from data.action4_direction_targets import prepare_evidence_batch
from model.action4_a0_enhancement import A0Enhancement,evidence_loss
from eval.action4_component_pipeline import load_trained_component
from model.training.action4_component_train import bounded,seed,config_id,rng_state,restore_rng
from model.training.action4_runtime import atomic_json,save_checkpoint_atomic


def build(cfg,device):
    base,parent_cfg=load_trained_component(bounded(cfg['base_checkpoint'],True),'A0',device)
    if parent_cfg['variant']!='baseline':raise ValueError('Original best heatmap A0 required')
    return A0Enhancement(base).to(device),parent_cfg['preprocess']


def publish(model,cfg,step,out):
    queue=out/'eval';queue.mkdir(exist_ok=True);request=queue/f'step_{step:08d}.json'
    if request.exists():return
    while sum(p.name.count('.')==1 and not p.with_suffix('.done.json').exists() for p in queue.glob('step_*.json'))>=8:
        if list(queue.glob('*.failed.json')):raise RuntimeError('Aux eval failed')
        time.sleep(5)
    snapshot=out/f'eval_step_{step:08d}.pt'
    save_checkpoint_atomic(snapshot,{'schema':'action4.aux_eval.v1','head':model.evidence.state_dict(),'config':cfg,'config_id':config_id(cfg),'step':step,'resumable':False})
    atomic_json(request,{'snapshot':str(snapshot.relative_to(R)),'step':step})


def train(config_path,resume=None):
    cfg=json.loads(bounded(config_path,True).read_text());out=bounded(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    for path,digest in cfg['source_identity'].items():
        if hashlib.sha256(bounded(path,True).read_bytes()).hexdigest()!=digest:raise ValueError('Changed auxiliary source: '+path)
    guard=(out/'training.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(2);seed(cfg['seed']);rng=np.random.default_rng(cfg['seed']+91);model,preprocess=build(cfg,'cuda:0')
    data=ComponentData(cfg['validation_receipt'],'train','A0');assert len(data)==3024
    optimizer=torch.optim.AdamW(model.trainable_parameters(),lr=cfg['lr'],weight_decay=.0001)
    total=cfg['epochs']*math.ceil(len(data)/cfg['batch'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,total,eta_min=cfg['lr']*.01)
    epoch=cursor=step=0
    if resume:
        state=torch.load(bounded(resume,True),map_location='cpu',weights_only=True)
        if state.get('schema')!='action4.aux_train.v1' or state['config_id']!=config_id(cfg):raise ValueError('Wrong recovery identity')
        model.evidence.load_state_dict(state['head'],strict=True);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);restore_rng(state['rng'],rng);epoch,cursor,step=state['epoch'],state['cursor'],state['step']
    elif (out/'last.pt').exists():raise FileExistsError('Explicit recovery required')
    atomic_json(out/'PROCESS.json',{'pid':os.getpid(),'role':'aux_train','base_frozen':True});publish(model,cfg,step,out)
    from torch.utils.data import DataLoader
    for ep in range(epoch,cfg['epochs']):
        order=np.random.default_rng(cfg['seed']+ep*1000003).permutation(len(data)).tolist();start=cursor if ep==epoch else 0
        batches=[order[i:i+cfg['batch']] for i in range(start,len(data),cfg['batch'])]
        loader=DataLoader(data,batch_sampler=batches,collate_fn=list,num_workers=cfg['workers'],generator=torch.Generator().manual_seed(cfg['seed']+ep))
        seen=start
        for items in loader:
            begin=time.monotonic();model.train();optimizer.zero_grad(set_to_none=True);record={}
            for offset in range(0,len(items),cfg['micro']):
                mini=items[offset:offset+cfg['micro']];x,points,mask,_=prepare_evidence_batch(mini,preprocess,'cuda:0',rng)
                losses=evidence_loss(model(x),points,mask);loss=sum(losses.values());weight=len(mini)/len(items)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite auxiliary loss')
                (loss*weight).backward()
                for key,value in losses.items():record[key]=record.get(key,0)+float(value.detach())*weight
            if any(p.grad is not None for p in model.base.parameters()):raise RuntimeError('Frozen base received a gradient')
            norm=torch.nn.utils.clip_grad_norm_(model.trainable_parameters(),1.,error_if_nonfinite=True);optimizer.step();scheduler.step();step+=1;seen+=len(items)
            next_epoch,next_cursor=(ep+1,0) if seen==len(data) else (ep,seen)
            status={'step':step,'epoch':ep+1,'cursor':seen,'losses':record,'gradient_norm':float(norm),'seconds':time.monotonic()-begin}
            atomic_json(out/'STATUS.json',status)
            with (out/'train.jsonl').open('a') as f:f.write(json.dumps(status)+'\n')
            if step%25==0 or next_cursor==0:
                save_checkpoint_atomic(out/'last.pt',{'schema':'action4.aux_train.v1','head':model.evidence.state_dict(),'config':cfg,'config_id':config_id(cfg),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'rng':rng_state(rng),'epoch':next_epoch,'cursor':next_cursor,'step':step,'resumable':True})
            if step%25==0:publish(model,cfg,step,out)
            if list((out/'eval').glob('*.failed.json')):raise RuntimeError('Auxiliary evaluation failed')
        cursor=0
    publish(model,cfg,step,out);atomic_json(out/'TRAIN_COMPLETE.json',{'step':step,'epochs':cfg['epochs']})

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--resume');a=p.parse_args()
    try:train(a.config,a.resume)
    except Exception:
        cfg=json.loads(bounded(a.config,True).read_text());out=bounded(cfg['output']);out.mkdir(parents=True,exist_ok=True);atomic_json(out/'FAILURE.json',{'traceback':traceback.format_exc()});raise
