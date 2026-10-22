"""Prepare sparse visible-pixel rays from immutable synthetic labels."""
from pathlib import Path
import json,numpy as np
from PIL import Image
from data import DataAdapter
from data.action4_dataset import virtual_roi
from data.action4_paths import PROJECT_ROOT as R,contained_path

def run():
    out=R/'outputs/action4_dense_surface_probe_v001';out.mkdir(exist_ok=True)
    records=[json.loads(l) for l in (R/'data/train/annotations.jsonl').read_text().splitlines()][:2];rows=[]
    y,x=np.meshgrid(np.arange(112),np.arange(112),indexing='ij');grid=np.stack((x,y),-1).reshape(-1,2)
    for record in records:
        for obj in record['instances']:
            if obj['ignore']:continue
            affine=DataAdapter.crop_affine(virtual_roi(obj['bbox_full_xyxy'],2.25),(112,112));native=DataAdapter.transform_points(grid,np.linalg.inv(affine));pixel=np.rint(native).astype(int)
            with Image.open(contained_path(R,obj['mask_visible'],exists=True)) as im:mask=np.asarray(im)
            valid=(pixel[:,0]>=0)&(pixel[:,1]>=0)&(pixel[:,0]<record['width'])&(pixel[:,1]<record['height']);indices=np.where(valid)[0];indices=indices[mask[pixel[indices,1],pixel[indices,0]]>0]
            rows.append({'sample_id':record['sample_id'],'instance_id':obj['instance_id'],'K':record['camera']['K'],'object_to_capture_cv_m':obj['canonical8_geometry']['object_to_capture_cv_m'],'native_pixels':native[indices].tolist(),'grid_indices':indices.tolist(),'mask_samples':len(indices),'scope':'derived synthetic mesh intersection, not measured depth'})
    (out/'INPUTS.json').write_text(json.dumps({'rows':rows}))
if __name__=='__main__':run()
