"""Independent component six-frame real review; demo is reference, never GT."""
import json,torch,numpy as np
from pathlib import Path
from PIL import Image,ImageDraw
from scipy.optimize import linear_sum_assignment
from model.training.action4_component_train import bounded
from model.training.action4_runtime import atomic_json
from eval.action4_component_pipeline import load_trained_component,predict_components
from eval.action4_video import draw_cuboids
from eval.action4_metrics import bbox_iou
from data.action4_paths import PROJECT_ROOT as R

def review(output):
 torch.set_num_threads(2);out=bounded(output);b,bc=load_trained_component(out/'B0.pt','B0','cpu');a,ac=load_trained_component(out/'A0.pt','A0','cpu')
 refs=json.loads((R/'outputs/action4_demo_alignment_v001/REFERENCE_OVERLAY_EXTRACTION.json').read_text())['frames'];manifest=json.loads((R/'data/action4_real/cache_v001/frames_manifest.json').read_text());rows=[]
 for ref in refs:
  i=ref['frame_id'];p=bounded(manifest['frames'][i]['image'],True)
  with Image.open(p) as im:rgb=im.convert('RGB');tensor=torch.from_numpy(np.asarray(rgb).copy()).permute(2,0,1)
  preds=predict_components(b,a,[tensor],bc,ac,'cpu')[0];small=rgb.resize((950,720));draw=ImageDraw.Draw(small);scale=np.array([950/rgb.width,720/rgb.height])
  for pred in preds:
   box=(np.array(pred['bbox_full_xyxy']).reshape(2,2)*scale).reshape(4).tolist();draw.rectangle(box,outline=(0,255,0),width=2);draw.text((box[0],box[1]),f"{pred['score']:.3f}",fill=(255,255,0))
  small.save(out/f'boxes_{i:08d}.png');draw_cuboids(rgb,preds,line_width=6).resize((950,720)).save(out/f'corners_{i:08d}.png')
  matrix=np.array([[bbox_iou((np.array(p['bbox_full_xyxy']).reshape(2,2)*scale).reshape(4),g['bbox_demo_px']) for g in ref['reference_overlay_components']] for p in preds]);pairs=[]
  if preds:
   x,y=linear_sum_assignment(-matrix);pairs=[float(matrix[u,v]) for u,v in zip(x,y)]
  rows.append(dict(frame_id=i,instances=preds,demo_reference_envelope_iou=pairs,both_reference_iou_ge05=len(pairs)==2 and min(pairs)>=.5));atomic_json(out/'STATUS.json',{'completed_frames':len(rows),'total':len(refs)})
 atomic_json(out/'REPORT.json',dict(scope='real-frame demo reference agreement, not GT metrics or final acceptance',frames=rows,both_reference_frames=sum(x['both_reference_iou_ge05'] for x in rows)))
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();review(a.output)
