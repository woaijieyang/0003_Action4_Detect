"""Same-checkpoint raw/refined A0 diagnostic, no training changes."""
import json,torch,numpy as np
from model.training.action4_component_train import bounded
from model.action4_component_experiments import Candidate
from model.action4_decode import decode_corners
from data.action4_component_data import roi_batch
from eval.action4_component_eval_cached import CachedComponentData
from model.training.action4_runtime import atomic_json

def audit(checkpoint,output):
 torch.set_num_threads(2);out=bounded(output);out.mkdir(parents=True,exist_ok=True)
 state=torch.load(bounded(checkpoint,True),map_location='cpu',weights_only=True);cfg=state['config'];step=state['step'];model=Candidate(cfg['role'],cfg['network'],cfg['variant']);model.load_state_dict(state['model']);del state;model.eval()
 data=CachedComponentData(cfg['validation_receipt'],'eval','A0');rows=[]
 with torch.inference_mode():
  for start in range(0,len(data),4):
   items=[data[i] for i in range(start,min(start+4,len(data)))];x,meta=roi_batch(items,cfg,'cpu',np.random.default_rng(0),False);result=model(x);raw=decode_corners(result['corner_logits']);refined=result['corners_hm']
   for j,g in enumerate(meta['gts']):
    a=(torch.linalg.vector_norm(raw[j]-meta['points_hm'][j],dim=-1)/meta['diagonal'][j]).tolist();b=(torch.linalg.vector_norm(refined[j]-meta['points_hm'][j],dim=-1)/meta['diagonal'][j]).tolist();rows.append(dict(sample_id=items[j]['record']['sample_id'],instance_id=g['instance_id'],raw_error=a,refined_error=b,visible=meta['visible'][j].tolist(),heavy=g.get('occlusion_inframe',0)>=.4,all_valid=bool(meta['valid'][j].all())))
   atomic_json(out/'STATUS.json',{'completed_rois':len(rows),'total':len(data),'checkpoint_step':step})
 def summary(key):
  return dict(all8_at05=sum(r['all_valid'] and max(r[key])<=.05 for r in rows),heavy_all8_at05=sum(r['heavy'] and r['all_valid'] and max(r[key])<=.05 for r in rows),hidden_mean=float(np.mean([e for r in rows for e,v in zip(r[key],r['visible']) if not v])),visible_mean=float(np.mean([e for r in rows for e,v in zip(r[key],r['visible']) if v])))
 atomic_json(out/'REPORT.json',dict(step=step,checkpoint=checkpoint,scope='all378 GT ROIs; same fixed weights raw versus correction; not full-system accuracy',raw=summary('raw_error'),refined=summary('refined_error'),per_roi=rows))
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args();audit(a.checkpoint,a.output)
