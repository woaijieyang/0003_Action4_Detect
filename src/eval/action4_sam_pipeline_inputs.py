"""Explicit predicted-SAM inputs for the existing staged inference pipeline.

This module never accepts GT and never invokes SAM. Frozen native predictions
are associated by image/instance/pixel domain, not array position.
"""
from dataclasses import dataclass
import hashlib
import json
import numpy as np
from data.action4_dataset import virtual_roi
from data.action4_sam_roi_inputs import (
    ImageFrame, verify_image_frame, prepare_sam_roi_inputs, RoiRequest,
)
from model.action4_sam_feature_adapter import FeatureIdentity, MaskAux

@dataclass(frozen=True)
class SamInferenceContext:
    frame: ImageFrame
    predictions: dict
    expected_sources: dict
    engineering_fixture: bool = False

def prepare_pipeline_sam(rgb, K, detections, context, *, roi_expansion=2.25):
    if not isinstance(context, SamInferenceContext):
        raise ValueError("SAM architecture requires explicit SamInferenceContext; missing masks use an empty prediction mapping")
    if type(context.engineering_fixture) is not bool:raise ValueError("Explicit engineering fixture boolean required")
    frame=verify_image_frame(context.frame)
    if not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.shape!=frame.pixels.shape or not np.array_equal(rgb,frame.pixels):
        raise ValueError("Inference RGB differs from SAM-bound native frame")
    from data.action4_roi_contract import validate_expansion
    roi_expansion=validate_expansion(roi_expansion)
    requests=[]
    seen=set()
    for i,d in enumerate(detections):
        pid=d.get("prediction_id")
        if not isinstance(pid,str) or not pid:
            raise ValueError("Explicit stable prediction ID required for native SAM association")
        if pid in seen:raise ValueError("Duplicate prediction ID in current frame")
        seen.add(pid)
        bounds=tuple(virtual_roi(d["bbox_full_xyxy"],roi_expansion))
        roi_id=hashlib.sha256(json.dumps({"image":frame.image_id,"prediction":pid,"bounds":bounds,"size":448},sort_keys=True).encode()).hexdigest()
        identity=FeatureIdentity(frame.image_id,pid,roi_id,"action4.roi448.v1")
        requests.append(RoiRequest(identity,0,bounds,K,frame.pixel_space_id))
    return prepare_sam_roi_inputs([frame],requests,context.predictions,
        expected_sources=context.expected_sources,mask_aux_factory=MaskAux,
        allow_synthetic=context.engineering_fixture)

def device_mask_aux(aux, device, dtype):
    return MaskAux(aux.visible.to(device=device,dtype=dtype),
        aux.distance.to(device=device,dtype=dtype),
        aux.available.to(device=device,dtype=dtype),
        aux.candidate_overlap.to(device=device,dtype=dtype),
        aux.dropout.to(device=device),aux.identities,aux.protocol_id)


class SamCacheContextProvider:
    """Prediction-only cached context; never reads item GT/instances."""
    def __init__(self,spec,*,split,expected_b0_sha256,allow_synthetic=False,validation_workers=1):
        from data.action4_sam_cache_provider import SamCacheProvider
        self.cache=SamCacheProvider(spec,split=split,allow_synthetic=allow_synthetic,validation_workers=validation_workers)
        if self.cache.sources["b0_model_sha256"]!=expected_b0_sha256:
            raise ValueError("SAM eval cache B0 differs from strict loaded bundle")

    def context(self,item):
        frame,predictions=self.cache.frame_predictions(item["record"]["sample_id"],item["image"])
        return SamInferenceContext(frame,predictions,self.cache.sources,self.cache.allow_synthetic)


def sam_eval_provider(pipeline,config,split):
    enhanced=pipeline.a0_kind=="joint_pose_sam"
    if not enhanced:
        if "sam_cache" in config:raise ValueError("Baseline eval cannot silently ignore SAM cache")
        return None
    if "sam_cache" not in config:raise ValueError("Enhanced eval requires explicit predicted SAM cache dependency")
    return SamCacheContextProvider(config["sam_cache"],split=split,
        expected_b0_sha256=pipeline.identity["B0_bundle_sha256"],
        allow_synthetic=config.get("engineering_fixture") is True,validation_workers=config.get("sam_cache_validation_workers",1))
