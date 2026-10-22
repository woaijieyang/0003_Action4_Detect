"""Pose RGB crop contract shared by jittered training and fixed detector-ROI validation."""
import numpy as np
import torch
from data import DataAdapter
from data.action4_dataset import prepare_rois

def prepare_pose_rgb(items,*,device="cpu",boxes=None,seeds=None,jitter_config=None):
    explicit_boxes=boxes is not None
    settings={"center_fraction":.15,"scale_range":[.65,1.35]} if jitter_config is None else jitter_config
    shift=float(settings["center_fraction"]);low,high=map(float,settings["scale_range"])
    if not (0<=shift and 0<low<=high and 2.25*low>=1+2*shift):
        raise ValueError("Jitter may not crop the target without updating visibility semantics")
    if boxes is not None and seeds is not None:raise ValueError("Detector ROI validation is never jittered")
    if seeds is not None and len(seeds)!=len(items):raise ValueError("One deterministic seed per target required")
    objects=[item["record"]["instances"][item["instance"]] for item in items]
    if boxes is None:
        boxes=[]
        for index,obj in enumerate(objects):
            box=np.asarray(obj["bbox_full_xyxy"],dtype=float);wh=box[2:]-box[:2];center=(box[:2]+box[2:])*.5
            if seeds is not None:
                rng=np.random.default_rng(int(seeds[index]))
                center=center+rng.uniform(-shift,shift,2)*wh
                wh=wh*rng.uniform(low,high)
            boxes.append(np.r_[center-wh*.5,center+wh*.5].tolist())
    if len(boxes)!=len(items):raise ValueError("Exact crop count required")
    rois,affines=prepare_rois([x["image"] for x in items],boxes,list(range(len(items))),
                              roi_size=448,expansion=2.25,device=device)
    down,_=DataAdapter.letterbox_affine((448,448),(112,112))
    geometries=[];metadata=[]
    for item,obj,box,affine in zip(items,objects,boxes,affines):
        record=item["record"];A=down@affine;K=A@np.asarray(record["camera"]["K"])
        T=np.asarray(obj["canonical8_geometry"]["object_to_capture_cv_m"])
        corners=DataAdapter.transform_points(obj["corners_px"],A)
        bounds=DataAdapter.transform_points(np.asarray(obj["bbox_full_xyxy"]).reshape(2,2),A)
        diagonal=float(np.linalg.norm(bounds[1]-bounds[0]))
        geometry={"K":K,"canonical":np.asarray(record["corners_3d_mm"])/1000,
                  "rotation":T[:3,:3],"translation":T[:3,3],"corners":corners,
                  "diagonal":diagonal,"visible":np.asarray(obj["corner_visible"],dtype=bool)}
        geometries.append(geometry)
        metadata.append({"sample_id":record["sample_id"],"instance_id":obj["instance_id"],
            "crop_box":box,"affine_hm":A.tolist(),"K_hm":K.tolist(),"canonical_m":geometry["canonical"].tolist(),
            "corners_hm":corners.tolist(),"visible":obj["corner_visible"],"diagonal":diagonal,
            "heavy":obj["occlusion_inframe"]>=.4,"crop_mode":"jitter" if seeds is not None else "explicit_prediction" if explicit_boxes else "GT"})
    targets={k:torch.as_tensor(np.asarray([g[k] for g in geometries]),device=device,dtype=torch.bool if k=="visible" else torch.float32) for k in geometries[0]}
    return rois,targets,metadata
