"""Ensure refined A0 coordinates, not discarded heatmaps, reach RGB-only output."""
from pathlib import Path
import sys,json,torch,numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from eval.action4_component_pipeline import predict_components
from data.action4_dataset import prepare_rois
from eval.action4_runner import heatmap_to_original
from model.action4_decode import transform_points
class B(torch.nn.Module):
 role='B0'
 def forward(self,x):
  z=torch.full((1,1,16,16),-20.);z[:,:,8,8]=10.
  return dict(center_logits=z,offset=torch.zeros(1,2,16,16),ltrb=torch.ones(1,4,16,16)*.2)
class A(torch.nn.Module):
 role='A0'
 def forward(self,x):return dict(corner_logits=torch.zeros(len(x),8,16,16),corners_hm=torch.arange(16).reshape(1,8,2).float().expand(len(x),-1,-1),visibility_logits=torch.zeros(len(x),8))
def main():
 torch.set_num_threads(1);cfg=dict(variant='baseline',preprocess=dict(full_size=[64,64],roi_size=64,roi_expansion=2.25),decode=dict(score_threshold=.3,topk=20,max_detections=2,duplicate_iou=.95,duplicate_center_ratio=.05,corner_topk=20));a_cfg=dict(cfg,variant='joint');images=[torch.zeros(3,64,64,dtype=torch.uint8)]
 rows=predict_components(B(),A(),images,cfg,a_cfg,'cpu');assert len(rows[0])==1
 _,aff=prepare_rois(images,[rows[0][0]['bbox_full_xyxy']],[0],roi_size=64,expansion=2.25,device='cpu')
 expected=transform_points(torch.arange(16).reshape(1,8,2).float(),torch.tensor(heatmap_to_original(aff,64,(16,16)),dtype=torch.float32))[0].tolist()
 assert rows[0][0]['corners_px']==expected and rows[0][0]['corner_ids']==list(range(8))
 (R/'outputs/action4_independent_trials_v001/PIPELINE_TEST.json').write_text(json.dumps(dict(status='PASS',refined_coordinates_reach_original_frame=True,all8_ids_preserved=True,scope='interface routing test, not recognition accuracy'),indent=2));print('PASS')
if __name__=='__main__':main()
