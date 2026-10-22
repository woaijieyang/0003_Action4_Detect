"""Joint GDR geometry inference; A0/B0 checkpoints stay independent."""
import json,hashlib
import numpy as np
import torch
from data import DataAdapter
from data.action4_paths import PROJECT_ROOT as R,contained_path
from data.action4_dataset import prepare_full_images,prepare_rois
from eval.action4_component_pipeline import load_trained_component
from model.action4_component_experiments import quality_decode
from model.action4_gdr_pose import GDRPose
from model.training.action4_runtime import canonical_hash

class GDRPosePipeline:
    def __init__(self,b0_checkpoint,a0_checkpoint,head_checkpoint,device="cuda:0"):
        self.device=device
        self.detector,self.b_cfg=load_trained_component(b0_checkpoint,"B0",device)
        self.base,self.a_cfg=load_trained_component(a0_checkpoint,"A0",device)
        cfg=json.loads((R/"configs/model/action4_gdr_pose_v001.json").read_text())
        cache_cfg=json.loads((R/"configs/model/action4_gdr_surface_v001.json").read_text())
        state=torch.load(contained_path(R,head_checkpoint,exists=True),map_location="cpu",weights_only=True)
        if state["config_id"]!=canonical_hash(cfg):raise ValueError("Joint pose checkpoint identity mismatch")
        if hashlib.sha256(contained_path(R,a0_checkpoint,exists=True).read_bytes()).hexdigest()!=cache_cfg["base_sha256"]:
            raise ValueError("Backbone differs from training parent")
        self.head=GDRPose().to(device);self.head.load_state_dict(state["head"],strict=True)
        self.head.eval();self.base.eval();self.detector.eval();self.head_step=state["step"]

    @torch.inference_mode()
    def predict(self,rgb,K,canonical):
        images,affines=prepare_full_images([rgb],size=tuple(self.b_cfg["preprocess"]["full_size"]),device=self.device)
        detections=quality_decode(self.detector(images),affines,self.b_cfg,self.b_cfg["variant"])[0]
        if not detections:return {"detections":[],"pose":[]}
        rois,crops=prepare_rois([rgb],[p["bbox_full_xyxy"] for p in detections],[0]*len(detections),roi_size=448,expansion=2.25,device=self.device)
        down,_=DataAdapter.letterbox_affine((448,448),(112,112))
        transforms=[down@a for a in crops]
        km=torch.tensor(np.stack([a@K for a in transforms]),dtype=torch.float32,device=self.device)
        cad=torch.tensor(np.array(canonical),dtype=torch.float32,device=self.device)[None].expand(len(detections),-1,-1)
        output=self.head(self.base.backbone(rois),km,cad)
        corners=output["corners"].cpu().numpy();camera=output["camera_corners"].cpu().numpy()
        result=[]
        for i,p in enumerate(detections):
            valid=bool(np.isfinite(corners[i]).all() and np.isfinite(camera[i]).all() and (camera[i,:,2]>0).all())
            result.append(dict(p,instance_id=i,object_id=1,corner_ids=list(range(8)),
                               corners_px=DataAdapter.transform_points(corners[i],np.linalg.inv(transforms[i])).tolist() if valid else None,
                               pose_valid=valid,scope="numeric validity is not accuracy"))
        return {"detections":detections,"pose":result}
