"""Async predicted-XYZ/mask -> PnP -> fixed-eight evaluation; no oracle inputs."""
import json,time,os,fcntl,traceback
import numpy as np
import torch
from data.action4_paths import PROJECT_ROOT as R
from data.action4_cached_surface_dataset import load_split
from model.action4_gdr_pose import GDRPose

from model.training.action4_runtime import atomic_json,canonical_hash


def evaluate(model,features,metadata):
    model.eval();rows=[]
    with torch.inference_mode():
        for start in range(0,len(features),16):
            ms=metadata[start:start+16]
            K=torch.tensor([m['K_hm'] for m in ms],device='cuda:0',dtype=torch.float32)
            canonical=torch.tensor([m['canonical_m'] for m in ms],device='cuda:0',dtype=torch.float32)
            output=model(features[start:start+16].to('cuda:0'),K,canonical)
            uv=output['corners'].cpu().numpy();camera=output['camera_corners'].cpu().numpy()
            for i,m in enumerate(ms):
                valid=bool(np.isfinite(uv[i]).all() and np.isfinite(camera[i]).all() and (camera[i,:,2]>0).all())
                pose={'valid':valid,'reason':None if valid else 'invalid_geometry'}
                error=(np.linalg.norm(uv[i]-m['corners_hm'],axis=1)/m['diagonal']).tolist() if valid else None
                rows.append({'sample_id':m['sample_id'],'instance_id':m['instance_id'],'valid_pose':pose['valid'],'reason':pose.get('reason'),'errors':error,'visible':m['visible'],'heavy':m['heavy'],'all8_at05':error is not None and max(error)<=.05})
    hidden=[e for row in rows if row['errors'] is not None for e,v in zip(row['errors'],row['visible']) if not v]
    return {'scope':'GT ROI joint GDR-ConvPnP diagnostic, not B0 or full-video accuracy','rois':len(rows),'all8_at05':sum(r['all8_at05'] for r in rows),'heavy_all8_at05':sum(r['heavy'] and r['all8_at05'] for r in rows),'heavy_count':sum(r['heavy'] for r in rows),'valid_poses':sum(r['valid_pose'] for r in rows),'hidden_mean_valid_only':float(np.mean(hidden)) if hidden else None,'invalid_poses_count_as_failures':True,'rows':rows}


def worker(config):
    cfg=json.loads((R/config).read_text());out=R/cfg['output'];lock=(out/'eval_worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);features,_,metadata=load_split(R/cfg['cache_output'],'eval');model=GDRPose().to('cuda:0');queue=out/'eval';queue.mkdir(exist_ok=True);atomic_json(out/'EVAL_PROCESS.json',{'pid':os.getpid()})
    def rank(report):return (report['all8_at05'],report['valid_poses'],-(report['hidden_mean_valid_only'] if report['hidden_mean_valid_only'] is not None else 1e9))
    while True:
        request=None
        for p in sorted(queue.glob('step_*.json')):
            if p.name.count('.')!=1 or p.with_suffix('.done.json').exists() or p.with_suffix('.failed.json').exists():continue
            try:fd=os.open(p.with_suffix('.claim'),os.O_EXCL|os.O_CREAT|os.O_WRONLY,0o600)
            except FileExistsError:continue
            with os.fdopen(fd,'w') as f:json.dump({'pid':os.getpid()},f)
            request=p;break
        if request is None:
            if (out/'TRAIN_COMPLETE.json').exists() or (out/'TRAIN_FAILURE.json').exists():return
            time.sleep(3);continue
        try:
            req=json.loads(request.read_text());state=torch.load(R/req['snapshot'],map_location='cpu',weights_only=True)
            if state['schema']!='action4.surface_eval.v1' or state['config_id']!=canonical_hash(cfg):raise ValueError('Eval identity mismatch')
            model.load_state_dict(state['head'],strict=True);report=evaluate(model,features,metadata);result={'step':req['step'],'report':report};atomic_json(request.with_suffix('.done.json'),result)
            bp=out/'BEST.json';best=json.loads(bp.read_text()) if bp.exists() else None
            if best is None or rank(report)>rank(best['report']):
                tmp=out/'best.tmp';os.link(R/req['snapshot'],tmp);os.replace(tmp,out/'best.pt');atomic_json(bp,result)
            (R/req['snapshot']).unlink()
        except BaseException:
            atomic_json(request.with_suffix('.failed.json'),{'traceback':traceback.format_exc()});raise
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();worker(a.config)
