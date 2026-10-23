"""Compose independently selected online B0 and joint GDR A0 checkpoints.

Detection threshold is frozen at0.3 as used by development operating metrics.
No ground-truth boxes, masks, poses or corners are accepted.
"""
import json,hashlib,os
import numpy as np
import torch
from data import DataAdapter
from data.action4_paths import PROJECT_ROOT as R,contained_path
from data.action4_dataset import prepare_rois
from model.action4_online_b0 import OnlineB0
from model.action4_gdr_pose import GDRPose
from eval.action4_online_b0_pipeline import predict as detect
from eval.action4_component_pipeline import load_trained_component
from model.training.action4_runtime import canonical_hash

class StagedPipeline:
    def __init__(self,b0_config,b0_checkpoint,a0_config,a0_checkpoint,a0_backbone,device="cuda:0",pose_method="network",a0_kind="joint_pose"):
        if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
            if torch.cuda.is_initialized():raise RuntimeError("Set CUBLAS_WORKSPACE_CONFIG before CUDA initialization")
            os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
        torch.use_deterministic_algorithms(True)
        self.device=device
        if pose_method not in ("network","pnp","network_iterative"):raise ValueError("Unknown pose decoder")
        self.pose_method=pose_method
        if a0_kind not in ("joint_pose","surface"):raise ValueError("Unknown A0 architecture")
        if a0_kind=="surface" and pose_method!="pnp":raise ValueError("Surface-only A0 requires explicit PnP decoding")
        self.a0_kind=a0_kind
        self.bcfg=json.loads(contained_path(R,b0_config,exists=True).read_text())
        self.acfg=json.loads(contained_path(R,a0_config,exists=True).read_text())
        bs=torch.load(contained_path(R,b0_checkpoint,exists=True),map_location="cpu",weights_only=True)
        az=torch.load(contained_path(R,a0_checkpoint,exists=True),map_location="cpu",weights_only=True)
        if bs["role"]!="B0" or bs["config_id"]!=canonical_hash(self.bcfg):raise ValueError("B0 identity mismatch")
        if az["schema"]!="action4.surface_eval.v1" or az["config_id"]!=canonical_hash(self.acfg):raise ValueError("A0 identity mismatch")
        cache=json.loads((R/"configs/model/action4_gdr_surface_v001.json").read_text())
        if hashlib.sha256(contained_path(R,a0_backbone,exists=True).read_bytes()).hexdigest()!=cache["base_sha256"]:
            raise ValueError("A0 backbone differs from selected head parent")
        self.b0=OnlineB0(self.bcfg["network"]).to(device);self.b0.load_state_dict(bs["model"],strict=True);self.b0.eval()
        self.base,_=load_trained_component(a0_backbone,"A0",device);self.base.eval()
        if a0_kind=="surface":
            from model.action4_surface_head import SurfaceHead
            self.a0=SurfaceHead().to(device)
        else:self.a0=GDRPose().to(device)
        self.a0.load_state_dict(az["head"],strict=True);self.a0.eval()
        self.identity={"B0_step":bs["step"],"B0_config_id":bs["config_id"],"A0_step":az["step"],"A0_kind":a0_kind,"A0_config_id":az["config_id"],"score_threshold":.3,"pose_method":pose_method,"deterministic_algorithms":True,"cublas_workspace":os.environ["CUBLAS_WORKSPACE_CONFIG"]}
        if {p.data_ptr() for p in self.b0.parameters()} & {p.data_ptr() for p in self.a0.parameters()}:
            raise ValueError("A0/B0 may not share parameter storage")


    @classmethod
    def from_bundles(cls,a0_bundle,b0_bundle,device="cuda:0"):
        from model.action4_release_bundle import load_bundle
        if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
            if torch.cuda.is_initialized(): raise RuntimeError("Set CUBLAS_WORKSPACE_CONFIG before CUDA initialization")
            os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
        torch.use_deterministic_algorithms(True)
        self=cls.__new__(cls);self.device=device
        self.b0,_,bs=load_bundle(b0_bundle,"B0",device)
        self.a0,self.base,az=load_bundle(a0_bundle,"A0",device)
        self.bcfg=bs["config"];self.acfg=az["config"]
        self.b0_scoring_policy=bs.get("inference_scoring_policy",self.bcfg.get("evaluation_scoring_policy","learned"))
        self.a0_kind=az["architecture"];self.pose_method=az["pose_method"]
        if self.a0_kind not in ("surface","joint_pose") or (self.a0_kind=="surface" and self.pose_method!="pnp"):
            raise ValueError("Invalid bundled architecture/decoder")
        self.identity={"B0_step":bs["selected_step"],"B0_config_id":bs["config_id"],"A0_step":az["selected_step"],
            "A0_kind":self.a0_kind,"A0_config_id":az["config_id"],"score_threshold":.3,"pose_method":self.pose_method,
            "deterministic_algorithms":True,"cublas_workspace":os.environ["CUBLAS_WORKSPACE_CONFIG"]}
        if self.b0_scoring_policy not in ("learned","center","product"):raise ValueError("Unknown bundled scoring policy")
        if self.b0_scoring_policy!="learned":self.identity["B0_scoring_policy"]=self.b0_scoring_policy
        if {p.data_ptr() for p in self.b0.parameters()} & {p.data_ptr() for p in self.a0.parameters()}:
            raise ValueError("Stages share parameter storage")
        return self

    def _detect(self,rgb):
        policy=getattr(self,"b0_scoring_policy","learned")
        if policy=="learned":return detect(self.b0,rgb,self.bcfg,self.device)
        from eval.action4_b0_scoring import predict
        return predict(self.b0,rgb,self.bcfg,self.device,policy)

    @torch.inference_mode()
    def predict(self,rgb,K,canonical,*,return_box_candidates=False):
        K=np.asarray(K,dtype=float);canonical=np.asarray(canonical,dtype=float)
        if K.shape!=(3,3) or canonical.shape!=(8,3) or not np.isfinite(K).all() or not np.isfinite(canonical).all():
            raise ValueError("Finite capture intrinsics and fixed eight CAD points required")
        box_candidates=self._detect(rgb)
        detections=[p for p in box_candidates if p["score"]>=.3]
        extra={"box_candidates":box_candidates} if return_box_candidates else {}
        if not detections:return dict(identity=self.identity,detections=[],instances=[],**extra)
        rois,affines=prepare_rois([rgb],[p["bbox_full_xyxy"] for p in detections],[0]*len(detections),
                                  roi_size=448,expansion=2.25,device=self.device)
        down,_=DataAdapter.letterbox_affine((448,448),(112,112));transforms=[down@a for a in affines]
        km=torch.tensor(np.stack([a@K for a in transforms]),device=self.device,dtype=torch.float32)
        cad=torch.tensor(canonical,device=self.device,dtype=torch.float32)[None].expand(len(detections),-1,-1)
        features=self.base.backbone(rois)
        if self.a0_kind=="surface":
            output={"surface":self.a0(features)};corners=camera=None
        else:
            output=self.a0(features,km,cad)
            corners=output["corners"].cpu().numpy();camera=output["camera_corners"].cpu().numpy()
        rows=[]
        for i,p in enumerate(detections):
            valid=corners is not None and bool(np.isfinite(corners[i]).all() and np.isfinite(camera[i]).all() and (camera[i,:,2]>0).all())
            diagnostic={"method":"network","valid":valid}
            uv=corners[i] if corners is not None else None
            if self.pose_method!="network":
                xyz=output["surface"]["xyz"][i].cpu().numpy()
                mask=output["surface"]["masks"][i,0].sigmoid().cpu().numpy()
                if self.pose_method=="pnp":
                    from model.action4_surface_pnp import decode_surface_pose
                    diagnostic=decode_surface_pose(xyz,mask,transforms[i]@K,canonical)
                else:
                    from model.action4_pose_refinement import refine_network_pose
                    diagnostic=refine_network_pose(xyz,mask,transforms[i]@K,canonical,
                        output["rotation"][i].cpu().numpy(),output["translation"][i].cpu().numpy())
                valid=diagnostic["valid"];uv=diagnostic["corners_hm"]
            rows.append(dict(p,instance_id=i,object_id=1,corner_ids=list(range(8)),
                             corners_px=DataAdapter.transform_points(uv,np.linalg.inv(transforms[i])).tolist() if valid else None,
                             pose_valid=valid,pose_diagnostic=diagnostic))
        return {"identity":self.identity,"detections":detections,"instances":rows,"scope":"numeric pose validity is not accuracy",**extra}


def load_pipeline(cfg,device="cuda:0"):
    pinned={}
    if "best_registry" in cfg:
        from model.action4_best_registry import verify_reference
        for role,key in [("A0","a0_bundle"),("B0","b0_bundle")]:
            pinned[role]=verify_reference(role,cfg[key],registry=cfg["best_registry"])
    if "a0_bundle" in cfg:
        pipeline=StagedPipeline.from_bundles(cfg["a0_bundle"],cfg["b0_bundle"],device)
    else:
        pipeline=StagedPipeline(cfg["b0_config"],cfg["b0_checkpoint"],cfg["a0_config"],cfg["a0_checkpoint"],cfg["a0_backbone"],device=device,pose_method=cfg["pose_method"],a0_kind=cfg.get("a0_kind","joint_pose"))
    policy=cfg.get("b0_scoring_policy",getattr(pipeline,"b0_scoring_policy","learned"))
    if policy not in ("learned","center","product"):raise ValueError("Unknown B0 scoring policy")
    pipeline.b0_scoring_policy=policy
    if policy!="learned":pipeline.identity["B0_scoring_policy"]=policy
    else:pipeline.identity.pop("B0_scoring_policy",None)
    for role,entry in pinned.items():
        if pipeline.identity[role+"_step"]!=entry["selected_step"] or pipeline.identity[role+"_config_id"]!=entry["config_id"]:
            raise ValueError("Loaded pipeline differs from selected best registry")
    return pipeline
