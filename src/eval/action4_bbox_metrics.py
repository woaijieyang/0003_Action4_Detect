"""B0 complete-body bbox metrics, independent of corner predictions."""
import numpy as np
from eval.action4_metrics import bbox_iou,maximum_matching

def score_boxes(predictions,references,score_threshold=.3):
 refs={r['sample_id']:r for r in references};preds={r['sample_id']:r for r in predictions}
 if len(refs)!=len(references) or len(preds)!=len(predictions) or set(refs)!=set(preds):raise ValueError('Exact unique image coverage required')
 rows=[]
 for sid,r in refs.items():
  gt=[g for g in r['instances'] if not g.get('ignore',False)];ps=[p for p in preds[sid]['instances'] if p['score']>=score_threshold];edges=[[j for j,g in enumerate(gt) if bbox_iou(p['bbox_full_xyxy'],g['bbox_full_xyxy'])>=.5] for p in ps];matches=maximum_matching(edges,len(gt));mp={p for p,g in matches};mg={g for p,g in matches};dups=sum(i not in mp and any(j in mg for j in edges[i]) for i in range(len(ps)))
  rows.append({'sample_id':sid,'scene':r.get('scene_family','unknown'),'tp':len(matches),'fp':len(ps)-len(matches),'fn':len(gt)-len(matches),'predictions':len(ps),'GT':len(gt),'matches':matches,'duplicate_predictions':dups,'both_detected':len(gt)==2 and len(matches)==2,'targetless':len(gt)==0,'targetless_false_positive':len(gt)==0 and len(ps)>0})
 def aggregate(xs):
  tp=sum(x['tp'] for x in xs);fp=sum(x['fp'] for x in xs);fn=sum(x['fn'] for x in xs);double=sum(x['GT']==2 for x in xs);empty=sum(x['targetless'] for x in xs);pred=sum(x['predictions'] for x in xs);duplicates=sum(x['duplicate_predictions'] for x in xs)
  return {'images':len(xs),'tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,'dual_images':double,'both_detected_images':sum(x['both_detected'] for x in xs),'dual_both_detection_rate':sum(x['both_detected'] for x in xs)/double if double else None,'duplicate_predictions':duplicates,'duplicate_rate_per_prediction':duplicates/pred if pred else None,'images_with_duplicates':sum(x['duplicate_predictions']>0 for x in xs),'targetless_images':empty,'targetless_false_positive_images':sum(x['targetless_false_positive'] for x in xs),'targetless_image_false_positive_rate':sum(x['targetless_false_positive'] for x in xs)/empty if empty else None}
 # AP uses the entire supplied confidence-ranked list, not the operating threshold.
 aps={};total=sum(sum(not g.get('ignore',False) for g in r['instances']) for r in refs.values());ranked=sorted([(float(p['score']),sid,i,p) for sid,r in preds.items() for i,p in enumerate(r['instances'])],key=lambda x:(-x[0],x[1],x[2]))
 for threshold in np.arange(.5,.951,.05):
  used={sid:set() for sid in refs};tp=[];fp=[]
  for score,sid,i,p in ranked:
   gs=[g for g in refs[sid]['instances'] if not g.get('ignore',False)];options=[(bbox_iou(p['bbox_full_xyxy'],g['bbox_full_xyxy']),j) for j,g in enumerate(gs) if j not in used[sid]];best=max(options,default=(-1,-1));ok=best[0]>=threshold-1e-9
   if ok:used[sid].add(best[1])
   tp.append(int(ok));fp.append(int(not ok))
  cum=np.cumsum(tp);precision=cum/np.maximum(1,cum+np.cumsum(fp));recall=cum/max(1,total);ap=float(np.mean([max(precision[recall>=t],default=0.) for t in np.linspace(0,1,101)])) if total else None;aps[f'{threshold:.2f}']=ap
 out={'schema':'action4.bbox_metrics_candidate.v1','scope':'B0 complete-body boxes only; no corner accuracy in primary score','operating_score_threshold':score_threshold,'matching':'IoU>=0.5 one-to-one maximum matching for operating metrics; confidence-greedy one-to-one matching for101point AP','AP_input_contract':'all supplied detections; caller must declare candidate cutoff/max detections','overall':aggregate(rows),'AP_by_IoU':aps,'mAP50':aps['0.50'],'mAP50_95':float(np.mean(list(aps.values()))) if total else None,'per_scene':{s:aggregate([x for x in rows if x['scene']==s]) for s in sorted({x['scene'] for x in rows})},'per_image':rows}
 strata={}
 byrow={x['sample_id']:x for x in rows}
 for sid,r in refs.items():
  valid=[g for g in r['instances'] if not g.get('ignore',False)];hit={g for p,g in byrow[sid]['matches']}
  for i,g in enumerate(valid):
   occ=g.get('occlusion_inframe');names=['occlusion/unknown' if occ is None else 'occlusion/lt0.1' if occ<.1 else 'occlusion/0.1to0.4' if occ<.4 else 'occlusion/ge0.4']
   box=g['bbox_full_xyxy'];ratio=(box[2]-box[0])/r['width'] if 'width' in r else None
   names.append('image_width_ratio/unknown' if ratio is None else 'image_width_ratio/lt0.06' if ratio<.06 else 'image_width_ratio/0.06to0.12' if ratio<.12 else 'image_width_ratio/ge0.12')
   if 'camera' in r and 'object_to_world_m' in g:
    transform=np.linalg.inv(np.asarray(r['camera']['camera_to_world_m']))@np.asarray(g['object_to_world_m']);depth=-float(transform[2,3]);names.append('synthetic_depth_m/lt0.7' if depth<.7 else 'synthetic_depth_m/0.7to1.0' if depth<1 else 'synthetic_depth_m/ge1.0')
   else:names.append('distance/unknown')
   for name in names:
    v=strata.setdefault(name,{'GT':0,'detected':0});v['GT']+=1;v['detected']+=i in hit
 for v in strata.values():v['recall']=v['detected']/v['GT']
 out['per_target_stratum']=strata;out['stratum_note']='Target-level detection recall; image width ratio is not measured distance; real distance unknown without calibration'
 return out
