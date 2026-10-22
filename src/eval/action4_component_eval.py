"""Shared async evaluation workers; each request scored once, never trains."""
import json,os,time,fcntl,traceback
from pathlib import Path
import numpy as np
import torch
from model.training.action4_component_train import bounded,config_id
from model.training.action4_runtime import atomic_json
from model.action4_component_experiments import Candidate,quality_decode
from model.action4_decode import decode_corners
from data.action4_component_data import ComponentData,roi_batch
from data.action4_dataset import prepare_full_images
from eval.action4_bbox_metrics import score_boxes
from data.action4_paths import PROJECT_ROOT as R

def evaluate(model,data,cfg,device):
 model.eval();rows=[];predictions=[];errors=[];hidden=[];visible=[];hits=[];heavy=[]
 with torch.inference_mode():
  batch_size=1 if cfg['role']=='B0' else 8
  for start in range(0,len(data),batch_size):
   items=[data[i] for i in range(start,min(start+batch_size,len(data)))]
   if cfg['role']=='B0':
    full,aff=prepare_full_images([x['image'] for x in items],size=tuple(cfg['preprocess']['full_size']),device=device)
    c=dict(cfg);c['decode']=dict(cfg['decode'],score_threshold=.001)
    ps=quality_decode(model(full),aff,c,cfg['variant'])
    predictions.extend(dict(sample_id=x['record']['sample_id'],instances=p) for x,p in zip(items,ps))
   else:
    rois,meta=roi_batch(items,cfg,device,np.random.default_rng(0),False);out=model(rois)
    uv=out['corners_hm'] if 'corners_hm' in out else decode_corners(out['corner_logits'])
    e=torch.linalg.vector_norm(uv-meta['points_hm'],dim=-1);normalized=e/meta['diagonal'][:,None];v=meta['visible'];valid=meta['valid']
    errors.extend(e[valid].cpu().tolist());hidden.extend(normalized[valid&~v].cpu().tolist());visible.extend(normalized[valid&v].cpu().tolist())
    for k,g in enumerate(meta['gts']):
     # Full8 success requires all8 projected domain points; no hidden-point dropping.
     ok=bool(valid[k].all() and (normalized[k]<=.05).all());hits.append(ok)
     if g.get('occlusion_inframe',0)>=.4:heavy.append(ok)
     rows.append(dict(sample_id=items[k]['record']['sample_id'],instance_id=g['instance_id'],normalized_errors=normalized[k].cpu().tolist(),visible=v[k].cpu().tolist(),all8_at05=ok))
 if cfg['role']=='B0':
  report=score_boxes(predictions,data.records,score_threshold=cfg['decode']['score_threshold']);report['AP_candidate_threshold']=.001;report['max_detections']=2
 else:report=dict(scope='GT ROI diagnostic, not full-system accuracy',rois=len(hits),all8_at05=sum(hits),heavy_count=len(heavy),heavy_all8_at05=sum(heavy),hidden_mean_normalized=float(np.mean(hidden)),visible_mean_normalized=float(np.mean(visible)),epe_mean_heatmap_px=float(np.mean(errors)),epe_p95_heatmap_px=float(np.percentile(errors,95)),per_roi=rows)
 report['dataset_role']='development_selection_not_independent_acceptance';return report

def rank(report,role):
 return (report['mAP50_95'],report['overall']['f1']) if role=='B0' else (report['all8_at05'],-report['hidden_mean_normalized'])

def worker(suite):
 torch.set_num_threads(1);settings=json.loads(bounded(suite,True).read_text());configs=[json.loads(bounded(p,True).read_text()) for p in settings['configs']]
 data={};model=None;last_key=None
 while True:
  claimed=None
  for cfg in configs:
   out=bounded(cfg['output']);q=out/'eval'
   for req in sorted(q.glob('step_*.json')):
    if req.name.count('.')!=1 or req.with_suffix('.done.json').exists() or req.with_suffix('.failed.json').exists():continue
    try:fd=os.open(req.with_suffix('.claim'),os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError:continue
    with os.fdopen(fd,'w') as f:json.dump({'pid':os.getpid(),'time':time.time()},f)
    claimed=(cfg,out,req);break
   if claimed:break
  if claimed is None:
   if all((bounded(c['output'])/'TRAIN_COMPLETE.json').exists() or (bounded(c['output'])/'FAILURE.json').exists() for c in configs):return
   time.sleep(3);continue
  cfg,out,req=claimed
  configs=configs[1:]+configs[:1]
  try:
   request=json.loads(req.read_text());state=torch.load(bounded(request['snapshot'],True),map_location='cpu',weights_only=True)
   if state['schema_version']!='action4.component_eval.v1' or state['config_id']!=config_id(cfg) or state['step']!=request['step']:raise ValueError('Eval snapshot identity mismatch')
   key=(cfg['role'],cfg['variant'])
   if key!=last_key:
    del model;model=Candidate(cfg['role'],cfg['network'],cfg['variant']).to('cuda:0');last_key=key
   model.load_state_dict(state['model'],strict=True);del state
   if cfg['role'] not in data:data[cfg['role']]=ComponentData(cfg['validation_receipt'],'eval',cfg['role'])
   report=evaluate(model,data[cfg['role']],cfg,'cuda:0');result={'step':request['step'],'snapshot':request['snapshot'],'report':report}
   atomic_json(req.with_suffix('.done.json'),result)
   with (out/'selection.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX);bp=out/'BEST.json';best=json.loads(bp.read_text()) if bp.exists() else None
    if best is None or rank(report,cfg['role'])>rank(best['report'],cfg['role']):
     link=out/f'best.{os.getpid()}.tmp';os.link(bounded(request['snapshot'],True),link);os.replace(link,out/'best.pt');atomic_json(bp,result)
    # Snapshot is complete and best.pt is an independent hardlink; preserve all metrics.
    bounded(request['snapshot'],True).unlink();fcntl.flock(lock,fcntl.LOCK_UN)
  except Exception:
   atomic_json(req.with_suffix('.failed.json'),{'error':traceback.format_exc()});raise


def recover_stale(suite):
 """Explicit recovery only; retain failure history and refuse live claim owners."""
 settings=json.loads(bounded(suite,True).read_text());recovered=[]
 for cp in settings['configs']:
  cfg=json.loads(bounded(cp,True).read_text());q=bounded(cfg['output'])/'eval'
  for claim in q.glob('step_*.claim'):
   req=claim.with_suffix('.json')
   if req.with_suffix('.done.json').exists():continue
   owner=json.loads(claim.read_text())
   try:os.kill(owner['pid'],0)
   except ProcessLookupError:pass
   else:continue
   suffix=str(time.time_ns());claim.rename(claim.with_name(claim.name+'.stale_'+suffix))
   failed=req.with_suffix('.failed.json')
   if failed.exists():failed.rename(failed.with_name(failed.name+'.history_'+suffix))
   recovered.append(str(req))
 return recovered

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--suite',required=True);p.add_argument('--recover-stale',action='store_true');a=p.parse_args()
 if a.recover_stale:print(recover_stale(a.suite))
 else:worker(a.suite)
