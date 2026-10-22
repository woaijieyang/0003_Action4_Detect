"""Async full-development eval of additive evidence against its frozen A0 base."""
import json,os,time,traceback
import torch,numpy as np
from model.training.action4_auxiliary_train import build
from model.training.action4_component_train import bounded,config_id
from model.training.action4_runtime import atomic_json
from data.action4_direction_targets import prepare_evidence_batch
from eval.action4_component_eval_cached import CachedComponentData
from model.action4_a0_enhancement import auxiliary_vote_evidence
from data import DataAdapter
from data.action4_paths import PROJECT_ROOT as R


def evaluate(model,data,preprocess):
    model.eval();rows=[]
    with torch.inference_mode():
        for start in range(0,len(data),4):
            items=[data[i] for i in range(start,min(start+4,len(data)))];x,points,mask,objects=prepare_evidence_batch(items,preprocess,'cuda:0');output=model(x)
            torch.manual_seed(20261022+start);torch.cuda.manual_seed_all(20261022+start)
            evidence=auxiliary_vote_evidence(output)
            for i,obj in enumerate(objects):
                # ROI transformation is uniform scaling; derive projected box diagonal from known ROI expansion.
                diagonal=(preprocess['roi_size']/4)/preprocess['roi_expansion']
                box=np.array(obj['bbox_full_xyxy']);extent=box[2:]-box[:2]
                diagonal*=np.linalg.norm(extent)/max(extent)
                base=torch.linalg.vector_norm(evidence['base_corners_hm'][i]-points[i],dim=-1)/diagonal
                aux=torch.linalg.vector_norm(evidence['auxiliary_corners_hm'][i]-points[i],dim=-1)/diagonal
                pred_mask=output['foreground_logits'][i].argmax(0).bool();gt_mask=mask[i].bool();union=(pred_mask|gt_mask).sum()
                rows.append({'sample_id':items[i]['record']['sample_id'],'instance_id':obj['instance_id'],'base_errors':base.cpu().tolist(),'auxiliary_errors':aux.cpu().tolist(),'visible':obj['corner_visible'],'heavy':obj['occlusion_inframe']>=.4,'valid_votes':evidence['numerically_valid'][i].cpu().tolist(),'mask_iou':float((pred_mask & gt_mask).sum()/union.clamp_min(1))})
    def aggregate(key):
        return {'all8_at05':sum(max(r[key])<=.05 for r in rows),'heavy_all8_at05':sum(r['heavy'] and max(r[key])<=.05 for r in rows),'hidden_mean':float(np.mean([e for r in rows for e,v in zip(r[key],r['visible']) if not v])),'visible_mean':float(np.mean([e for r in rows for e,v in zip(r[key],r['visible']) if v]))}
    return {'scope':'GT ROI diagnostic; numerical-validity fallback candidate, not production fusion','rois':len(rows),'base':aggregate('base_errors'),'auxiliary':aggregate('auxiliary_errors'),'recoveries':sum(max(r['base_errors'])>.05 and max(r['auxiliary_errors'])<=.05 for r in rows),'regressions':sum(max(r['base_errors'])<=.05 and max(r['auxiliary_errors'])>.05 for r in rows),'valid_vote_points':sum(sum(r['valid_votes']) for r in rows),'mean_mask_iou':float(np.mean([r['mask_iou'] for r in rows])),'per_roi':rows,'production_accepted':False}


def worker(config_path):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);cfg=json.loads(bounded(config_path,True).read_text());out=bounded(cfg['output']);queue=out/'eval';queue.mkdir(parents=True,exist_ok=True);model,preprocess=build(cfg,'cuda:0');data=CachedComponentData(cfg['validation_receipt'],'eval','A0')
    atomic_json(out/'EVAL_PROCESS.json',{'pid':os.getpid(),'gpu':1})
    while True:
        request=None
        for p in sorted(queue.glob('step_*.json')):
            if p.name.count('.')!=1 or p.with_suffix('.done.json').exists() or p.with_suffix('.failed.json').exists():continue
            try:fd=os.open(p.with_suffix('.claim'),os.O_EXCL|os.O_CREAT|os.O_WRONLY,0o600)
            except FileExistsError:continue
            with os.fdopen(fd,'w') as f:json.dump({'pid':os.getpid(),'time':time.time()},f)
            request=p;break
        if request is None:
            if (out/'TRAIN_COMPLETE.json').exists() or (out/'FAILURE.json').exists():return
            time.sleep(3);continue
        try:
            req=json.loads(request.read_text());state=torch.load(bounded(req['snapshot'],True),map_location='cpu',weights_only=True)
            if state['schema']!='action4.aux_eval.v1' or state['config_id']!=config_id(cfg):raise ValueError('Auxiliary checkpoint identity mismatch')
            model.evidence.load_state_dict(state['head'],strict=True);del state;report=evaluate(model,data,preprocess);result={'step':req['step'],'snapshot':req['snapshot'],'report':report};atomic_json(request.with_suffix('.done.json'),result)
            best_path=out/'BEST.json';best=json.loads(best_path.read_text()) if best_path.exists() else None
            rank=lambda r:(r['auxiliary']['all8_at05'],-r['auxiliary']['hidden_mean'])
            if best is None or rank(report)>rank(best['report']):
                temp=out/'best.tmp';os.link(bounded(req['snapshot'],True),temp);os.replace(temp,out/'best.pt');atomic_json(best_path,result)
            bounded(req['snapshot'],True).unlink()
        except Exception:
            atomic_json(request.with_suffix('.failed.json'),{'error':traceback.format_exc()});raise

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();worker(a.config)
