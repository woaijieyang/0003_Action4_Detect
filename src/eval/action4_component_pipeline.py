"""RGB-only independent B0 -> ROI -> A0 pipeline, including refined coordinates."""
import torch,numpy as np
from pathlib import Path
from data.action4_paths import PROJECT_ROOT,contained_path
from data.action4_dataset import prepare_full_images,prepare_rois
from model.action4_component_experiments import Candidate,quality_decode
from model.action4_decode import decode_corners,transform_points
from eval.action4_runner import heatmap_to_original

def load_trained_component(path,role,device):
 state=torch.load(contained_path(PROJECT_ROOT,Path(path),exists=True),map_location='cpu',weights_only=True)
 if state.get('schema_version') not in ('action4.component_eval.v1','action4.component_training.v1') or state.get('role')!=role:raise ValueError('Independent trained role checkpoint required')
 cfg=state['config'];model=Candidate(role,cfg['network'],cfg['variant']);model.load_state_dict(state['model'],strict=True)
 return model.to(device).eval(),cfg

def predict_components(detector,corners,images,b_cfg,a_cfg,device):
 if detector.role!='B0' or corners.role!='A0':raise ValueError('Component roles reversed')
 detector.eval();corners.eval()
 with torch.inference_mode():
  full,full_aff=prepare_full_images(images,size=tuple(b_cfg['preprocess']['full_size']),device=device)
  rows=quality_decode(detector(full),full_aff,b_cfg,b_cfg['variant'])
  ids=[i for i,row in enumerate(rows) for p in row];boxes=[p['bbox_full_xyxy'] for row in rows for p in row]
  if not boxes:return rows
  size=a_cfg['preprocess']['roi_size'];rois,aff=prepare_rois(images,boxes,ids,roi_size=size,expansion=a_cfg['preprocess']['roi_expansion'],device=device)
  output=corners(rois);hm=output['corner_logits'];uv=output['corners_hm'] if 'corners_hm' in output else decode_corners(hm,topk=a_cfg['decode']['corner_topk'])
  matrix=torch.as_tensor(heatmap_to_original(aff,size,(hm.shape[-1],hm.shape[-2])),device=device,dtype=uv.dtype)
  original=transform_points(uv,matrix).cpu().tolist();peaks=hm.sigmoid().flatten(2).amax(-1).cpu().tolist()
  visibility=output['visibility_logits'].sigmoid().cpu().tolist() if 'visibility_logits' in output else None
  k=0
  for row in rows:
   for i,p in enumerate(row):
    p.update(instance_id=i,object_id=1,corner_ids=list(range(8)),corners_px=original[k],corner_peak_scores=peaks[k],corner_method=a_cfg['variant'],score_policy='center_times_quality' if b_cfg['variant']=='giou_quality' else 'center',identity_scope='frame-local detection index, not physical left/right identity')
    if visibility is not None:p['predicted_corner_visibility']=visibility[k]
    k+=1
  return rows
