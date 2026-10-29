"""Pose RGB crop contract shared by jittered training and fixed detector-ROI validation."""
import numpy as np
import torch
from data import DataAdapter
from data.action4_dataset import prepare_rois

def prepare_pose_rgb(items,*,device="cpu",boxes=None,seeds=None,jitter_config=None,crop_specs=None,roi_expansion=2.25,roi_source=None):
    from data.action4_roi_contract import validate_expansion
    roi_expansion=validate_expansion(roi_expansion)
    predicted=roi_source=="frozen_b0_prediction"
    if predicted:
        if boxes is None or seeds is not None or jitter_config is not None or crop_specs is not None:
            raise ValueError("Frozen prediction crops require only explicit predicted boxes, no fallback or jitter")
    elif roi_source is not None or roi_expansion!=2.25:
        raise ValueError("Nonlegacy expansion requires explicit frozen prediction source")
    crop_metadata=None
    if crop_specs is not None:
        if boxes is not None or seeds is not None or jitter_config is not None:
            raise ValueError("Resolved crop_specs cannot be combined with boxes/seeds/jitter settings")
        from data.action4_training_crop_specs import validate_crop_specs
        boxes,crop_metadata=validate_crop_specs(items,crop_specs)
    explicit_boxes=boxes is not None
    settings={"center_fraction":.15,"scale_range":[.65,1.35]} if jitter_config is None else jitter_config
    shift=float(settings["center_fraction"]);low,high=map(float,settings["scale_range"])
    if not predicted and not (0<=shift and 0<low<=high and 2.25*low>=1+2*shift):
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
    rois,affines,roi_specs=prepare_rois([x["image"] for x in items],boxes,list(range(len(items))),
                              roi_size=448,expansion=roi_expansion,device=device,return_specs=True)
    down,_=DataAdapter.letterbox_affine((448,448),(112,112))
    geometries=[];metadata=[]
    for item,obj,box,affine,spec in zip(items,objects,boxes,affines,roi_specs):
        record=item["record"];A=down@affine;K=A@np.asarray(record["camera"]["K"])
        T=np.asarray(obj["canonical8_geometry"]["object_to_capture_cv_m"])
        if not np.isfinite(T).all() or not np.isfinite(K).all() or not np.isfinite(np.asarray(obj["corners_px"])).all():raise ValueError("Invalid GT geometry before projection")
        corners=DataAdapter.transform_points(obj["corners_px"],A)
        bounds=DataAdapter.transform_points(np.asarray(obj["bbox_full_xyxy"]).reshape(2,2),A)
        diagonal=float(np.linalg.norm(bounds[1]-bounds[0]))
        geometry={"K":K,"canonical":np.asarray(record["corners_3d_mm"])/1000,
                  "rotation":T[:3,:3],"translation":T[:3,3],"corners":corners,
                  "diagonal":diagonal,"visible":np.asarray(obj["corner_visible"],dtype=bool)}
        geometries.append(geometry)
        metadata.append({"sample_id":record["sample_id"],"instance_id":obj["instance_id"],
            "roi_expansion":roi_expansion,"roi_source":roi_source,"roi_spec":spec.record(),"crop_box":box,"affine_hm":A.tolist(),"K_hm":K.tolist(),"canonical_m":geometry["canonical"].tolist(),
            "corners_hm":corners.tolist(),"visible":obj["corner_visible"],"diagonal":diagonal,
            "heavy":obj["occlusion_inframe"]>=.4,"crop_mode":"jitter" if seeds is not None else "explicit_prediction" if explicit_boxes else "GT"})
        # Loss-only native observations; never reuse112 projected coordinates as native.
        corner_down,_=DataAdapter.letterbox_affine((448,448),(224,224))
        corner_A=corner_down@affine
        camera_corners=geometry["canonical"]@T[:3,:3].T+T[:3,3]
        corner_bounds=DataAdapter.transform_points(np.asarray(obj["bbox_full_xyxy"]).reshape(2,2),corner_A)
        from data.action4_dataset import is_ignored
        metadata[-1]["corner_gt224"]={
            "schema":"action4.training_corner_gt224.v1","uv_space":"native_integer_centers",
            "image_id":record["sample_id"],"gt_instance_id":str(obj["instance_id"]),
            "native_uv":np.asarray(obj["corners_px"],dtype=float).tolist(),
            "physical_depth_m":camera_corners[:,2].tolist(),
            "A224_from_native":corner_A.tolist(),
            "full_box_diagonal224":float(np.linalg.norm(corner_bounds[1]-corner_bounds[0])),
            "gt_known":list(obj.get("corner_annotation_known",[True]*8)),
            "ignore":[bool(is_ignored(obj))]*8}
        if crop_metadata is not None:
            metadata[-1].update(crop_mode="crop_spec",crop_spec=crop_metadata[len(metadata)-1])
    targets={k:torch.as_tensor(np.asarray([g[k] for g in geometries]),device=device,dtype=torch.bool if k=="visible" else torch.float32) for k in geometries[0]}
    return rois,targets,metadata
