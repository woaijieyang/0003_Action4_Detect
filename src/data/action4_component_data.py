"""Shared one-time accepted-data validation and stat-checked component loaders."""
import json,hashlib
from pathlib import Path
import numpy as np
from data.action4_paths import PROJECT_ROOT,contained_path
from data.action4_dataset import Action4Dataset,is_ignored,prepare_rois,build_corner_targets
from data import DataAdapter

def prepare_validation(path):
 path=contained_path(PROJECT_ROOT,Path(path),exists=False)
 if path.exists():return
 out={'schema':'action4.component_data_validation.v1','splits':{}}
 for split in ('train','eval'):
  manifest=PROJECT_ROOT/f'data/{split}/annotations.jsonl'
  ds=Action4Dataset(PROJECT_ROOT,manifest,split=split,validation_workers=12)
  out['splits'][split]={'manifest':str(manifest.relative_to(PROJECT_ROOT)),'sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'stats':ds._validated_stats}
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(out))

class ComponentData:
 def __init__(self,receipt,split,role):
  receipt=contained_path(PROJECT_ROOT,Path(receipt),exists=True);entry=json.loads(receipt.read_text())['splits'][split]
  manifest=contained_path(PROJECT_ROOT,Path(entry['manifest']),exists=True)
  if hashlib.sha256(manifest.read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('Manifest changed')
  self.records=DataAdapter(PROJECT_ROOT).read_samples(manifest,check_images=False);self.stats=entry['stats'];self.role=role
  if any(r['status']!='training_accepted' or r['split']!=split for r in self.records):raise ValueError('Accepted split required')
  self.units=[(i,j) for i,r in enumerate(self.records) for j,g in enumerate(r['instances']) if not is_ignored(g)] if role=='A0' else [(i,None) for i in range(len(self.records))]
 def __len__(self):return len(self.units)
 def __getitem__(self,index):
  import torch
  from PIL import Image
  i,j=self.units[index];r=self.records[i];p=contained_path(PROJECT_ROOT,Path(r['image']),exists=True)
  if tuple(self.stats[r['image']])!=Action4Dataset._file_identity(p):raise ValueError('Validated image identity changed')
  with Image.open(p) as im:array=np.asarray(im.convert('RGB')).copy()
  return {'image':torch.from_numpy(array).permute(2,0,1).contiguous(),'record':r,'instance':j}

def roi_batch(items,cfg,device,rng,jitter):
 import torch
 def _jitter_box(box,rng,enabled,center_fraction,scale_range):
  if not enabled:return box
  box=np.asarray(box);size=box[2:]-box[:2];center=(box[:2]+box[2:])/2+rng.uniform(-center_fraction,center_fraction,2)*size;size=size*rng.uniform(*scale_range)
  return np.r_[center-size/2,center+size/2].tolist()
 images=[x['image'] for x in items];records=[x['record'] for x in items];gts=[r['instances'][x['instance']] for x,r in zip(items,records)]
 boxes=[_jitter_box(g['bbox_full_xyxy'],rng,enabled=jitter,center_fraction=.15,scale_range=(.65,1.35)) for g in gts];ids=list(range(len(items)));gis=[x['instance'] for x in items]
 size=cfg['preprocess']['roi_size'];rois,aff=prepare_rois(images,boxes,ids,roi_size=size,expansion=cfg['preprocess']['roi_expansion'],device=device)
 targets,valid=build_corner_targets(records,ids,gis,aff,roi_size=size,heatmap_size=size//4,device=device)
 down,_=DataAdapter.letterbox_affine((size,size),(size//4,size//4));points=[DataAdapter.transform_points(g['corners_px'],down@a) for g,a in zip(gts,aff)]
 # Visibility is a label only; never an input to the model.
 visible=[g['corner_visible'] for g in gts]
 diag=[np.linalg.norm(DataAdapter.transform_points(np.array(g['bbox_full_xyxy']).reshape(2,2),down@a)[1]-DataAdapter.transform_points(np.array(g['bbox_full_xyxy']).reshape(2,2),down@a)[0]) for g,a in zip(gts,aff)]
 meta=dict(targets=targets,valid=valid,roi_batch_index=torch.arange(len(items),device=device),batch_size=len(items),points_hm=torch.as_tensor(np.array(points),device=device,dtype=torch.float32),visible=torch.as_tensor(np.array(visible),device=device,dtype=torch.bool),diagonal=torch.tensor(diag,device=device).float().clamp_min(1),gts=gts)
 return rois,meta
