"""Versioned CAD-correspondence supervision; pose/XYZ never network inputs."""
from pathlib import Path
import json
import numpy as np
import torch
from PIL import Image
from data import DataAdapter
from data.action4_paths import PROJECT_ROOT as R,contained_path
from data.action4_dataset import prepare_rois


def regions():
    p=R/'data/action4_surface_regions_v001/centers.json'
    if not p.exists():
        points=np.load(R/'data/DJI_Action4_construct/r3_v001/mesh_v001.npz')['vertices_mm']/1000
        distance=np.full(len(points),np.inf);index=int(np.argmax(np.linalg.norm(points-points.mean(0),axis=1)));chosen=[]
        for _ in range(64):
            chosen.append(index);distance=np.minimum(distance,np.sum((points-points[index])**2,axis=1));index=int(np.argmax(distance))
        p.parent.mkdir(exist_ok=True)
        with p.open('x') as f:json.dump({'method':'deterministic farthest point centers on frozen CAD vertices','xyz_object_m':points[chosen].tolist(),'vertex_indices':chosen,'regions':64},f,indent=2)
    return np.array(json.loads(p.read_text())['xyz_object_m'])


def prepare_surface_batch(items,preprocess,split,device='cpu'):
    if split not in ['train','eval','test']:raise ValueError('Unknown split')
    review=json.loads((R/'outputs/action4_surface_labels_v002/ROOT_VISUAL_REVIEW.json').read_text())
    if not review['candidate_training_allowed']:raise ValueError('Surface labels not admitted')
    size=preprocess['roi_size'];assert size==448 and preprocess['roi_expansion']==2.25
    objects=[x['record']['instances'][x['instance']] for x in items];boxes=[g['bbox_full_xyxy'] for g in objects]
    images,affines=prepare_rois([x['image'] for x in items],boxes,list(range(len(items))),roi_size=size,expansion=2.25,device=device)
    centers=regions();xyz_targets=[];valid_targets=[];region_targets=[];mask_targets=[];y,x=np.meshgrid(np.arange(112),np.arange(112),indexing='ij');grid=np.stack((x,y),-1).reshape(-1,2);down,_=DataAdapter.letterbox_affine((448,448),(112,112))
    for item,obj,affine in zip(items,objects,affines):
        record=item['record'];assert record['split']==split
        name=record['sample_id']+'_i'+str(obj['instance_id']);path=contained_path(R,R/f'data/{split}/surface_v002_candidate/{name}.npz',exists=True)
        with np.load(path) as d:points=d['xyz_object_m'].astype(float);idx=d['hit_grid_indices'];missing=d['missing_grid_indices'];native=d['native_pixels']
        projected=DataAdapter.transform_points(native,down@affine);assert np.allclose(projected,np.c_[idx%112,idx//112],atol=1e-4,rtol=0)
        canonical=np.array(record['corners_3d_mm'])/1000;minimum=canonical.min(0);extent=canonical.max(0)-minimum
        xyz=np.zeros((3,112*112),np.float32);xyz[:,idx]=((points-minimum)/extent).T;valid=np.zeros(112*112,bool);valid[idx]=True
        region=np.full(112*112,-100,np.int64);region[idx]=np.argmin(np.sum((points[:,None]-centers[None])**2,axis=-1),axis=1)+1
        visible=np.zeros(112*112,np.float32);visible[np.r_[idx,missing]]=1
        native_all=DataAdapter.transform_points(grid,np.linalg.inv(down@affine));pixel=np.rint(native_all-np.array(obj['amodal_expanded_origin_px'])).astype(int)
        with Image.open(contained_path(R,obj['mask_amodal_expanded'],exists=True)) as im:
            assert im.size==tuple(obj['amodal_expanded_shape_hw'][::-1]);amodal_image=np.asarray(im)
            inside=(pixel[:,0]>=0)&(pixel[:,1]>=0)&(pixel[:,0]<im.width)&(pixel[:,1]<im.height);amodal=np.zeros(112*112,np.float32);amodal[inside]=(amodal_image[pixel[inside,1],pixel[inside,0]]>0).astype(np.float32)
        xyz_targets.append(xyz.reshape(3,112,112));valid_targets.append(valid.reshape(112,112));region_targets.append(region.reshape(112,112));mask_targets.append(np.stack([visible,amodal]).reshape(2,112,112))
    targets={k:torch.as_tensor(np.array(v),device=device) for k,v in [('xyz',xyz_targets),('valid',valid_targets),('regions',region_targets),('masks',mask_targets)]}
    return images,targets
