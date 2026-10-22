"""Prepare versioned surface-label rays without changing published RGB/labels."""
import json
import numpy as np
from PIL import Image
from data import DataAdapter
from data.action4_dataset import virtual_roi
from data.action4_paths import PROJECT_ROOT as R,contained_path
from model.training.action4_runtime import atomic_json

def run():
    out=R/'outputs/action4_surface_labels_v001';assert json.loads((out/'MESH_BINDING.json').read_text())['status']=='PASS'
    inputs=out/'inputs';inputs.mkdir(exist_ok=False);rows=[]
    y,x=np.meshgrid(np.arange(112),np.arange(112),indexing='ij');grid=np.stack((x,y),-1).reshape(-1,2)
    for split in ['train','eval','test']:
        destination=contained_path(R,R/f'data/{split}/surface_v001_candidate',exists=False);destination.mkdir(exist_ok=False)
        records=[json.loads(l) for l in (R/f'data/{split}/annotations.jsonl').read_text().splitlines()]
        for record in records:
            for obj in record['instances']:
                if obj['ignore']:continue
                name=record['sample_id']+'_i'+str(obj['instance_id']);affine=DataAdapter.crop_affine(virtual_roi(obj['bbox_full_xyxy'],2.25),(112,112));native=DataAdapter.transform_points(grid,np.linalg.inv(affine));pixel=np.rint(native).astype(int)
                with Image.open(contained_path(R,obj['mask_visible'],exists=True)) as im:mask=np.asarray(im)
                valid=(pixel[:,0]>=0)&(pixel[:,1]>=0)&(pixel[:,0]<record['width'])&(pixel[:,1]<record['height']);idx=np.where(valid)[0];idx=idx[mask[pixel[idx,1],pixel[idx,0]]>0]
                source=inputs/(name+'.npz')
                with source.open('xb') as f:np.savez_compressed(f,native_pixels=native[idx],grid_indices=idx.astype(np.int32),K=np.array(record['camera']['K']),object_to_capture=np.array(obj['canonical8_geometry']['object_to_capture_cv_m']))
                rows.append({'split':split,'sample_id':record['sample_id'],'instance_id':obj['instance_id'],'input':str(source.relative_to(R)),'output':str((destination/(name+'.npz')).relative_to(R)),'mask_samples':len(idx),'occlusion':obj['occlusion_inframe']})
            if len(rows)%100<2:atomic_json(out/'INPUT_PROGRESS.json',{'prepared_objects':len(rows)})
    assert len(rows)==3802
    atomic_json(out/'INPUTS_COMPLETE.json',{'rows':rows,'objects':len(rows),'source':'immutable2000 published manifests','RGB_changed':False,'training_accepted':False})
if __name__=='__main__':run()
