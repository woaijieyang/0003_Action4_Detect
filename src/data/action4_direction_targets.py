"""Visible-object surface labels for an additive direction-field branch."""
import numpy as np
import torch
from PIL import Image
from data import DataAdapter
from data.action4_paths import PROJECT_ROOT,contained_path
from data.action4_dataset import prepare_rois


def prepare_evidence_batch(items, preprocess, device="cpu", rng=None):
    size=preprocess["roi_size"]
    images=[item["image"] for item in items]
    objects=[item["record"]["instances"][item["instance"]] for item in items]
    boxes=[obj["bbox_full_xyxy"] for obj in objects]
    if rng is not None:
        perturbed=[]
        for box in boxes:
            box=np.asarray(box);extent=box[2:]-box[:2]
            centre=(box[:2]+box[2:])/2+rng.uniform(-.15,.15,2)*extent
            extent=extent*rng.uniform(.65,1.35)
            perturbed.append(np.r_[centre-extent/2,centre+extent/2].tolist())
        boxes=perturbed
    rois,affines=prepare_rois(images,boxes,list(range(len(items))),roi_size=size,
                             expansion=preprocess["roi_expansion"],device=device)
    down,_=DataAdapter.letterbox_affine((size,size),(size//4,size//4))
    h=size//4
    y,x=np.meshgrid(np.arange(h),np.arange(h),indexing="ij")
    grid=np.stack((x,y),axis=-1).reshape(-1,2)
    points=[]
    masks=[]
    for obj,affine in zip(objects,affines):
        matrix=down@affine
        points.append(DataAdapter.transform_points(obj["corners_px"],matrix))
        pixel=np.rint(DataAdapter.transform_points(grid,np.linalg.inv(matrix))).astype(np.int64)
        with Image.open(contained_path(PROJECT_ROOT,obj["mask_visible"],exists=True)) as im:
            native=np.asarray(im)
        inside=(pixel[:,0]>=0)&(pixel[:,0]<native.shape[1])&(pixel[:,1]>=0)&(pixel[:,1]<native.shape[0])
        mask=np.zeros(len(grid),dtype=bool)
        mask[inside]=native[pixel[inside,1],pixel[inside,0]]>0
        masks.append(mask.reshape(h,h))
    return rois,torch.tensor(np.array(points),device=device,dtype=torch.float32),torch.tensor(np.array(masks),device=device),objects
