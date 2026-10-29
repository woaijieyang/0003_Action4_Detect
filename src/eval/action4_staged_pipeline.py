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
        from eval.action4_surface_snapshot import bind_model_file,check_model_file
        self._surface_model_bindings={role:bind_model_file(path) for role,path in
            (("B0",b0_checkpoint),("A0",a0_checkpoint),("A0_backbone",a0_backbone))}
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
        for binding in self._surface_model_bindings.values():check_model_file(binding)
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
        from model.action4_checkpoint_paths import resolve_checkpoint
        from eval.action4_surface_snapshot import bind_model_file,check_model_file,model_identities
        self._surface_model_bindings={role:bind_model_file(resolve_checkpoint(path)) for role,path in
                                      (("B0",b0_bundle),("A0",a0_bundle))}
        self.b0,_,bs=load_bundle(self._surface_model_bindings["B0"]["path"],"B0",device)
        self.a0,self.base,az=load_bundle(self._surface_model_bindings["A0"]["path"],"A0",device)
        for binding in self._surface_model_bindings.values():check_model_file(binding)
        self.bcfg=bs["config"];self.acfg=az["config"]
        from data.action4_roi_contract import validate_expansion
        self.roi_expansion=validate_expansion(self.acfg.get("roi_expansion",2.25))
        self.b0_scoring_policy=bs.get("inference_scoring_policy",self.bcfg.get("evaluation_scoring_policy","learned"))
        self.a0_kind=az["architecture"];self.pose_method=az["pose_method"]
        self.geometry_batch_policy="instancewise_head_v1"
        if self.a0_kind not in ("surface","joint_pose","joint_pose_sam") or (self.a0_kind=="surface" and self.pose_method!="pnp"):
            raise ValueError("Invalid bundled architecture/decoder")
        self.identity={"B0_step":bs["selected_step"],"B0_config_id":bs["config_id"],"A0_step":az["selected_step"],
            "A0_kind":self.a0_kind,"A0_config_id":az["config_id"],"score_threshold":.3,"pose_method":self.pose_method,
            "deterministic_algorithms":True,"cublas_workspace":os.environ["CUBLAS_WORKSPACE_CONFIG"]}
        if self.a0_kind=="joint_pose_sam":
            self.identity["B0_bundle_sha256"]=model_identities(self._surface_model_bindings)["B0"]["sha256"]
            self.identity["sam_protocol_id"]=az["sam_protocol"]["protocol_id"]
        if self.b0_scoring_policy not in ("learned","center","product"):raise ValueError("Unknown bundled scoring policy")
        if self.b0_scoring_policy!="learned":self.identity["B0_scoring_policy"]=self.b0_scoring_policy
        if {p.data_ptr() for p in self.b0.parameters()} & {p.data_ptr() for p in self.a0.parameters()}:
            raise ValueError("Stages share parameter storage")
        return self

    @classmethod
    def from_corner_bundles(cls,a0_bundle,b0_bundle,eval_sam_cache,device="cpu",*,engineering_fixture=False,full_box_quality_receipt=None):
        """Explicit research-only corner path; default deployment loader stays closed."""
        from eval.action4_corner_prediction import create_corner_pipeline
        return create_corner_pipeline(cls,a0_bundle,b0_bundle,eval_sam_cache,
            device=device,engineering_fixture=engineering_fixture,full_box_quality_receipt=full_box_quality_receipt)

    @classmethod
    def from_sealed_corner(cls,seal_binding,device="cpu"):
        from eval.action4_corner_selected_inference import create_selected_pipeline
        return create_selected_pipeline(cls,seal_binding,device)

    def predict_sealed_corner(self,rgb,K,canonical,*,sam_context):
        from eval.action4_corner_selected_inference import predict_selected_corner
        return predict_selected_corner(self,rgb,K,canonical,sam_context=sam_context)

    def _detect_corner(self,rgb):
        if not getattr(self,"corner_candidate_mode",False):raise ValueError("Explicit corner B0 diagnostic path required")
        from eval.action4_b0_scoring import predict
        return predict(self.b0,rgb,self.bcfg,self.device,self.b0_scoring_policy,include_refined_quality=True)

    def corner_context(self,image_id,rgb):
        if not getattr(self,"corner_candidate_mode",False):raise ValueError("Corner candidate context requires explicit constructor")
        from eval.action4_corner_prediction import cache_context
        return cache_context(self,image_id,rgb)

    def predict_corner(self,rgb,K,canonical,*,sam_context,diagnostic_candidate=False,engineering_route_gates=None):
        if not getattr(self,"corner_candidate_mode",False):raise ValueError("Explicit corner candidate constructor required")
        from eval.action4_corner_prediction import predict_corner
        return predict_corner(self,rgb,K,canonical,sam_context=sam_context,diagnostic_candidate=diagnostic_candidate,engineering_route_gates=engineering_route_gates)

    def _detect(self,rgb):
        policy=getattr(self,"b0_scoring_policy","learned")
        if policy=="learned":return detect(self.b0,rgb,self.bcfg,self.device)
        from eval.action4_b0_scoring import predict
        return predict(self.b0,rgb,self.bcfg,self.device,policy)

    @torch.inference_mode()
    def predict_video(self,rgb,K,canonical,*,video_context,return_surface=False,surface_frame=None,roi_expansion=None,sam_video_context=None,pose_stabilizer=None,frame_clock=None,motion_observations=None,corner_trial_binding=None):
        if corner_trial_binding is not None:
            if pose_stabilizer is not None or frame_clock is not None or motion_observations is not None:
                raise ValueError("Corner video trial cannot silently combine another postprocessor")
            if surface_frame is not None and (sam_video_context is None or surface_frame is not sam_video_context.frame):
                raise ValueError("Corner surface export must use actual current SAM/RGB frame")
            if getattr(self,"roi_expansion",None) not in (1.5,1.15):raise ValueError("Loaded selected corner ROI required")
            if roi_expansion is not None and roi_expansion!=self.roi_expansion:
                raise ValueError("Corner video ROI must equal the selected model ROI")
            from eval.action4_corner_video import predict_corner_video
            return predict_corner_video(self,rgb,K,canonical,video_context=video_context,
                sam_video_context=sam_video_context,trial_binding=corner_trial_binding,return_surface=return_surface)
        if self.a0_kind=="joint_pose_sam":
            from eval.action4_video_sam_inputs import VideoSamFeatureContext
            if not isinstance(sam_video_context,VideoSamFeatureContext):
                raise ValueError("Enhanced video requires authenticated current SAM features")
        elif sam_video_context is not None:
            raise ValueError("Baseline video cannot silently ignore mask features")
        if pose_stabilizer is not None:
            from model.action4_video_pose_stabilizer import VideoPoseStabilizer
            if not isinstance(pose_stabilizer,VideoPoseStabilizer) or not isinstance(frame_clock,dict) or frame_clock.get("frame_id")!=video_context.frame_id:
                raise ValueError("Explicit video stabilizer and matching frame/PTS clock required")
        elif frame_clock is not None or motion_observations is not None:raise ValueError("Frame stabilization clock/evidence without consumer")
        result=self.predict(rgb,K,canonical,_video_context=video_context,sam_context=sam_video_context,
            return_surface=return_surface,surface_frame=surface_frame,roi_expansion=roi_expansion)
        if pose_stabilizer is not None:
            current_supports=None
            if pose_stabilizer.config.require_reference_shape:
                if sam_video_context is None:raise ValueError("Reference-shape gate needs authenticated current SAM masks")
                masks,receipt=sam_video_context.consume(rgb,K,video_context,result["detections"],
                    b0_sha256=self.identity["B0_bundle_sha256"])
                if receipt["frame_id"]!=frame_clock["frame_id"] or receipt["pts"]!=frame_clock["pts"]:
                    raise ValueError("Reference mask clock differs")
                current_supports=dict(frame_id=frame_clock["frame_id"],pts=frame_clock["pts"],
                    time_base=frame_clock["time_base"],masks=masks)
            result=pose_stabilizer.apply_prediction(result,frame_id=frame_clock["frame_id"],pts=frame_clock["pts"],
                time_base=frame_clock["time_base"],K_native=K,canonical_m=canonical,mode="video",
                motion_observations=motion_observations,current_supports=current_supports)
        return result

    @torch.inference_mode()
    def predict(self,rgb,K,canonical,*,return_box_candidates=False,sam_context=None,return_surface=False,surface_frame=None,_video_context=None,roi_expansion=None,frozen_detections=None):
        if getattr(self,"corner_candidate_mode",False):raise ValueError("Unadmitted corner candidate requires explicit predict_corner; deployment predict refused")
        from data.action4_roi_contract import validate_expansion
        roi_expansion=validate_expansion(getattr(self,"roi_expansion",2.25) if roi_expansion is None else roi_expansion)
        K=np.asarray(K,dtype=float);canonical=np.asarray(canonical,dtype=float)
        if K.shape!=(3,3) or canonical.shape!=(8,3) or not np.isfinite(K).all() or not np.isfinite(canonical).all():
            raise ValueError("Finite capture intrinsics and fixed eight CAD points required")
        enhanced=self.a0_kind=="joint_pose_sam"
        if type(return_surface) is not bool:raise ValueError("Explicit return_surface boolean required")
        if not return_surface and surface_frame is not None:raise ValueError("Unused surface_frame requires return_surface=True")
        if return_surface and (self.pose_method!="network" or self.a0_kind not in ("joint_pose","joint_pose_sam")):
            raise ValueError("Surface snapshots currently require raw network joint pose; PnP/refined/surface-only unsupported")
        if enhanced:
            from eval.action4_sam_pipeline_inputs import SamInferenceContext
            if _video_context is None:
                if not isinstance(sam_context,SamInferenceContext):raise ValueError("Explicit predicted SAM image context required")
            else:
                from eval.action4_video_sam_inputs import VideoSamFeatureContext
                if not isinstance(sam_context,VideoSamFeatureContext):raise ValueError("Explicit verified video SAM feature context required")
                expected_sam=self.acfg.get("sam_cache",{}).get("expected_sources",{}).get("sam_model_sha256")
                if expected_sam is None or sam_context.sam_model_sha256!=expected_sam:
                    raise ValueError("Video SAM weights differ from the enhanced model input recipe")
            if sam_context.expected_sources.get("b0_model_sha256")!=self.identity.get("B0_bundle_sha256"):
                raise ValueError("SAM prompt B0 dependency differs from the loaded B0 bundle")
        elif sam_context is not None:
            raise ValueError("Baseline A0 cannot silently ignore SAM inputs")
        export_frame=None;surface_models=None
        if return_surface:
            from eval.action4_surface_snapshot import checked_frame,model_identities
            export_rgb=(np.moveaxis(rgb.detach().cpu().numpy() if hasattr(rgb,"detach") else np.asarray(rgb),0,-1) if _video_context is not None else rgb)
            export_frame=checked_frame(surface_frame if surface_frame is not None else (sam_context.frame if enhanced else None),export_rgb)
            if enhanced:
                checked_frame(sam_context.frame,export_rgb)
                if (export_frame.image_id,export_frame.pixel_space_id,export_frame.source_sha256,export_frame.variant_id,export_frame.variant_sha256)!=(sam_context.frame.image_id,sam_context.frame.pixel_space_id,sam_context.frame.source_sha256,sam_context.frame.variant_id,sam_context.frame.variant_sha256):
                    raise ValueError("Surface/SAM source identity mismatch")
            surface_models=model_identities(getattr(self,"_surface_model_bindings",None))
        if _video_context is not None and frozen_detections is not None:
            raise ValueError("Video resolved ROI cannot be overridden by raw detection replay")
        if _video_context is None:
            if frozen_detections is None:box_candidates=self._detect(rgb)
            else:
                from data.action4_frozen_detection_context import FrozenDetectionContext
                from eval.action4_surface_snapshot import model_identities
                if not isinstance(frozen_detections,FrozenDetectionContext):raise ValueError("Typed frozen detections required")
                sources=model_identities(self._surface_model_bindings)
                box_candidates=frozen_detections.consume(rgb,K,b0_sha256=sources["B0"]["sha256"],scoring_policy=self.b0_scoring_policy)
            if any("track_id" in p or "temporal_status" in p or "sam_support_admission_id" in p
                   or p.get("origin")=="SAM_tracking_geometry" or p.get("B0_current_observation") is False
                   for p in box_candidates):
                raise ValueError("Temporal predictions require an explicit video-only context")
        else:
            from data.action4_video_roi_context import VideoROIContext
            from eval.action4_surface_snapshot import model_identities
            if not isinstance(_video_context,VideoROIContext):
                raise ValueError("Explicit video ROI context required")
            sources=model_identities(self._surface_model_bindings)
            box_candidates=_video_context.consume(rgb,K,b0_sha256=sources["B0"]["sha256"])
        if enhanced or return_surface:
            # Bind BEFORE score filtering. Filtered list position is not identity.
            stable=[]
            for candidate in box_candidates:
                candidate=dict(candidate)
                if return_surface and _video_context is not None:
                    if candidate.get("prediction_id")!=f"video:{candidate['track_id']}:frame:{_video_context.frame_id}":raise ValueError("Video surface track/frame identity differs")
                elif return_surface:
                    grid=candidate.get("grid_index")
                    if not isinstance(grid,(int,np.integer)) or isinstance(grid,bool) or grid<0:
                        raise ValueError("Surface export requires actual stable B0 grid_index")
                    expected=f"b0:{export_frame.image_id}:grid:{int(grid)}"
                    if "prediction_id" in candidate and candidate["prediction_id"]!=expected:
                        raise ValueError("Surface detector prediction/grid identity mismatch")
                    candidate["prediction_id"]=expected
                if "prediction_id" not in candidate:
                    grid=candidate.get("grid_index")
                    if not isinstance(grid,(int,np.integer)) or isinstance(grid,bool) or grid<0:
                        raise ValueError("Enhanced detector requires stable prediction_id or grid_index")
                    candidate["prediction_id"]=f"b0:{sam_context.frame.image_id}:grid:{int(grid)}"
                stable.append(candidate)
            if len({p["prediction_id"] for p in stable})!=len(stable):
                raise ValueError("Duplicate native detector prediction ID")
            box_candidates=stable
        if _video_context is None:
            detections=[p for p in box_candidates if p["score"]>=.3]
        else:
            detections=_video_context.select_for_a0(box_candidates)
            if self.a0_kind not in ("joint_pose","joint_pose_sam") and any(p.get("temporal_status")=="sam_supported" for p in detections):
                raise ValueError("SAM geometry ROI requires baseline joint-pose video; surface/enhanced need separate admission")
        extra={"box_candidates":box_candidates} if return_box_candidates else {}
        extra["roi_expansion"]=roi_expansion
        if return_surface:extra["surface_snapshots"]=[]
        if _video_context is not None:
            extra["video_roi_consumption"]={"frame_id":_video_context.frame_id,
                "replay_identity":_video_context.replay_identity,
                "sam_role":"SAM_tracking_geometry_to_RGB_ROI" if any(p.get("temporal_status")=="sam_supported" for p in detections) else "B0_association_only",
                "A0_mask_features_enabled":enhanced,"withheld":json.loads(_video_context.withheld_json),
                "selection_policy":{"observed_B0_min_score":.3,
                    "sam_supported":"explicit_current_mask_geometry_admission_no_B0_score",
                    "predicted_or_lost":"withheld"}}

        if not detections:return dict(identity=self.identity,detections=[],instances=[],**extra)
        sam_inputs=None
        if enhanced:
            if _video_context is None:
                from eval.action4_sam_pipeline_inputs import prepare_pipeline_sam
                sam_inputs=prepare_pipeline_sam(rgb,K,detections,sam_context,roi_expansion=roi_expansion)
            else:
                from eval.action4_video_sam_inputs import prepare_video_pipeline_sam
                sam_inputs,roi_specs=prepare_video_pipeline_sam(rgb,K,detections,sam_context,video_context=_video_context,
                    b0_sha256=self.identity["B0_bundle_sha256"],roi_expansion=roi_expansion)
            rois=sam_inputs.rgb.to(self.device);transforms=list(sam_inputs.A112);affines=list(sam_inputs.A448)
        else:
            sampling_device="cpu" if getattr(self,"roi_sampling_device","model")=="cpu" else self.device
            rois,affines,roi_specs=prepare_rois([rgb],[p["bbox_for_a0"] if _video_context is not None else p["bbox_full_xyxy"] for p in detections],[0]*len(detections),
                                      roi_size=448,expansion=roi_expansion,device=sampling_device,return_specs=True)
            rois=rois.to(self.device)
            down,_=DataAdapter.letterbox_affine((448,448),(112,112));transforms=[down@a for a in affines]
        if not enhanced or _video_context is not None:
            extra["a0_roi_specs"]=[dict(spec.record(),prediction_id=p.get("prediction_id"),
                track_id=p.get("track_id"),roi_source=p.get("roi_source",p.get("origin","B0_prediction")),
                A448=spec.affine().tolist(),A112=spec.affine(112).tolist(),K112=spec.camera(K,112).tolist())
                for spec,p in zip(roi_specs,detections)]
        km=torch.tensor(np.stack([a@K for a in transforms]),device=self.device,dtype=torch.float32)
        if _video_context is not None:
            extra["video_roi_consumption"]["instances"]=[
                {"track_id":p["track_id"],"bbox_used":p["bbox_for_a0"],"roi_decision_id":p["roi_decision_id"],
                 "origin":p.get("origin",p.get("box_origin")),
                 "B0_current_observation":p.get("B0_current_observation",p.get("temporal_status")=="observed"),
                 "sam_support_admission_id":p.get("sam_support_admission_id"),
                 "consumption_path":"current_SAM_support->estimated_body_bbox->RGB_ROI448->A112_K112->baseline_A0" if p.get("temporal_status")=="sam_supported" else "current_B0->tracked_body_bbox->RGB_ROI448->A112_K112->baseline_A0",
                 "A448":np.asarray(affines[i]).tolist(),"A112":np.asarray(transforms[i]).tolist(),
                 "K112":km[i].cpu().tolist()} for i,p in enumerate(detections)]
        if enhanced and _video_context is not None:
            for item in extra["video_roi_consumption"]["instances"]:
                item["consumption_path"]="current_resolved_video_ROI+current_native_mask->same_RGB_MaskAux_K->enhanced_C5->A0"
            extra["video_roi_consumption"]["SAM_feature_receipt"]=json.loads(sam_context.receipt_json)
        cad=torch.tensor(canonical,device=self.device,dtype=torch.float32)[None].expand(len(detections),-1,-1)
        features=self.base.backbone(rois)
        if self.a0_kind=="surface":
            output={"surface":self.a0(features)};corners=camera=None
        else:
            from model.action4_geometry_forward import instancewise_geometry
            if enhanced:
                from eval.action4_sam_pipeline_inputs import device_mask_aux
                mask_aux=device_mask_aux(sam_inputs.mask_aux,self.device,rois.dtype)
                output,sam_output=self.a0(features,km,cad,rgb=rois,mask_aux=mask_aux,
                    identities=mask_aux.identities,return_aux=True)
            else:
                output=instancewise_geometry(self.a0,features,km,cad)
            corners=output["corners"].cpu().numpy();camera=output["camera_corners"].cpu().numpy()
        if return_surface:
            from eval.action4_surface_snapshot import make_snapshot
            extra["surface_snapshots"]=[make_snapshot(frame=export_frame,prediction=p,output=output,index=i,
                A112=transforms[i],A448=affines[i],K_native=K,K112=km[i],canonical=cad[i],
                models=surface_models,sam_input=sam_inputs.source_dependencies[i] if enhanced else None,video_context=_video_context)
                for i,p in enumerate(detections)]
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
            sam_trace={}
            if enhanced:
                sam_trace["sam_input"]=dict(available=bool(sam_inputs.mask_aux.available[i].item()),
                    reason=sam_inputs.reasons[i],dependencies=sam_inputs.source_dependencies[i],
                    q_mask=float(sam_output.q_mask[i].item()),q_mask_is_calibrated=False,
                    residual_l2=float(sam_output.residual[i].float().norm().item()),
                    consumption_path="native_mask->same_RGB_ROI->enhanced_C5->surface_pose->canonical8")
            pid=p.get("prediction_id",f"b0:{i}")
            rows.append(dict(p,instance_id=i,object_id=1,corner_ids=list(range(8)),
                             prediction_id=pid,source_detection_ids=p.get("source_detection_ids",[pid]),published=True,positive_depth_verified=bool(valid),
                             corners_px=DataAdapter.transform_points(uv,np.linalg.inv(transforms[i])).tolist() if valid else None,
                             pose_valid=valid,pose_diagnostic=diagnostic,**sam_trace))
        if "rotation" in output and "translation" in output:
            for i,row in enumerate(rows):
                r=output["rotation"][i].detach().cpu().numpy();t=output["translation"][i].detach().cpu().numpy()
                finite=bool(np.isfinite(r).all() and np.isfinite(t).all())
                row["raw_network_pose"]={"coordinate_frame":"object_to_capture_cv_m","finite":finite,
                    "rotation":r.tolist() if finite else None,"translation_m":t.tolist() if finite else None}
        if _video_context is not None:
            for row in rows:
                row["corners_raw_from_resolved_roi"]=json.loads(json.dumps(row["corners_px"])) if self.pose_method=="network" else None
                row["corners_final"]=json.loads(json.dumps(row["corners_px"]))
                row["post_A0_geometry"]="none" if self.pose_method=="network" else "existing_decoder_requires_separate_admission"
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
    from data.action4_roi_contract import validate_expansion
    pipeline.roi_expansion=validate_expansion(cfg.get("roi_expansion",getattr(pipeline,"roi_expansion",2.25)))
    if "roi_sampling_device" in cfg:
        sampling=cfg["roi_sampling_device"]
        if sampling not in ("cpu","model") or (pipeline.a0_kind=="joint_pose_sam" and sampling!="cpu"):
            raise ValueError("Explicit ROI sampling backend differs from the implemented SAM/training path")
        pipeline.roi_sampling_device=sampling;pipeline.identity["ROI_sampling_device"]=sampling
    return pipeline
