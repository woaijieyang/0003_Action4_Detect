"""Immutable predicted-video-mask bridge to existing SAM A0 forward inputs.

No SAM/model execution, GT fallback, pose production or quality certification.
Expected/current binding must originate from an authoritative frame/track/session
ledger. Equality of caller-supplied bindings is not physical identity proof.
VideoROIContext lacks PTS: this bridge binds declared PTS, not independent native
frame-timestamp verification. Caller supplies existing factories. Each prepare
returns fresh owned tensors; the snapshot itself stores only immutable scalars,
tuples and bytes. The root must bind current track/replay identity at consumption.
"""
from dataclasses import dataclass
import hashlib
import json
import numpy as np
from .action4_video_roi_context import VideoROIContext, rgb_digest
from .action4_dataset import virtual_roi
from .action4_sam_roi_inputs import (
    ImageFrame, NativeMaskIdentity, NativeSamPrediction, RoiRequest,
    array_digest, native_key, prediction_metadata, prepare_sam_roi_inputs, _sources,
)

def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

def _sha(value):
    if not isinstance(value,str) or len(value)!=64 or value=="0"*64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Explicit SHA256 required")

def _text(value):
    if not isinstance(value,str) or not value.strip():raise ValueError("Explicit identity required")

@dataclass(frozen=True)
class VideoMaskBinding:
    frame_id: int
    pts: int
    time_base: tuple
    pixel_space_id: str
    track_id: str
    track_generation: int
    native_obj_id: int
    mask_generation: int
    session_id: str
    replay_identity: str
    mask_sha256: str | None
    a0_selection_sha256: str

    def validate(self):
        for v in (self.frame_id,self.track_generation,self.native_obj_id,self.mask_generation):
            if type(v) is not int or v<0:raise ValueError("Nonnegative frame/object/generation required")
        if type(self.pts) is not int:raise ValueError("Integer PTS required")
        if (type(self.time_base) is not tuple or len(self.time_base)!=2
                or any(type(v) is not int or v<=0 for v in self.time_base)):
            raise ValueError("Positive rational time base required")
        for v in (self.pixel_space_id,self.track_id,self.session_id):_text(v)
        _sha(self.replay_identity);_sha(self.a0_selection_sha256)
        if self.mask_sha256 is not None:_sha(self.mask_sha256)

@dataclass(frozen=True)
class VideoSamSnapshot:
    binding: VideoMaskBinding
    context_json: str
    native_rgb_shape: tuple
    native_rgb_bytes: bytes
    mask_bytes: bytes | None
    status: str
    reason: str | None
    candidate_overlap: float
    sources_json: str
    quality_evidence_json: str
    roi_expansion: float = 2.25
    roi_output_size: int = 448

    @classmethod
    def capture(cls,context,rgb,K,*,expected_binding,prediction_binding,mask,
                status,reason,expected_sources,candidate_overlap=0.0,quality_evidence=None,
                roi_expansion=2.25,roi_output_size=448):
        _roi_policy(roi_expansion,roi_output_size)
        if not isinstance(context,VideoROIContext):raise ValueError("Existing VideoROIContext required")
        if not isinstance(expected_binding,VideoMaskBinding) or not isinstance(prediction_binding,VideoMaskBinding):
            raise ValueError("Explicit expected and predicted video binding required")
        expected_binding.validate();prediction_binding.validate()
        if expected_binding!=prediction_binding:raise ValueError("Swapped/stale mask binding")
        b=expected_binding
        if b.frame_id!=context.frame_id or b.replay_identity!=context.replay_identity:
            raise ValueError("Replay/frame binding differs")
        _sources(expected_sources)
        if expected_sources["b0_model_sha256"]!=context.b0_sha256:raise ValueError("B0 source differs")
        detections=context.consume(rgb,K,b0_sha256=context.b0_sha256)
        chosen=[d for d in detections if d["track_id"]==b.track_id]
        if len(chosen)!=1:raise ValueError("One current observed project track required")
        a=rgb.detach().cpu().numpy() if hasattr(rgb,"detach") else np.asarray(rgb)
        # consume has already checked native CHW uint8 identity.
        hwc=np.ascontiguousarray(a.transpose(1,2,0))
        allowed={"available","absent","missing_response","inference_error","session_lost","target_visibility_unknown","rejected"}
        if status not in allowed:raise ValueError("Explicit mask status required")
        if status!="available" and (not isinstance(reason,str) or not reason):
            raise ValueError("Unavailable/absent evidence requires reason")
        if not isinstance(candidate_overlap,(int,float)) or not np.isfinite(candidate_overlap) or not 0<=candidate_overlap<=1:
            raise ValueError("Invalid candidate overlap")
        if status=="absent" and mask is not None:raise ValueError("Absent prediction cannot carry mask bytes")
        payload=None
        if mask is not None:
            m=np.asarray(mask)
            if m.dtype!=np.bool_ or m.shape!=hwc.shape[:2]:raise ValueError("Native bool mask must match RGB space")
            if array_digest(m)!=b.mask_sha256:raise ValueError("Native mask identity differs")
            if status!="available" and m.any():raise ValueError("Unavailable mask cannot contain foreground")
            payload=m.tobytes(order="C")
        elif b.mask_sha256 is not None:raise ValueError("Declared mask bytes missing")
        if status=="available" and mask is None:raise ValueError("Available response requires actual mask")
        evidence={"status":"unknown"} if quality_evidence is None else quality_evidence
        if not isinstance(evidence,dict):raise ValueError("Quality evidence must be explicit metadata")
        # Retain evidence only; this bridge never certifies predicted mask quality.
        context_data={"frame_id":context.frame_id,"rgb_identity":context.rgb_identity,
            "b0_sha256":context.b0_sha256,"replay_identity":context.replay_identity,
            "K_native":list(context.K_native),"detection":chosen[0]}
        return cls(b,_json(context_data),tuple(hwc.shape),hwc.tobytes(),payload,status,reason,
                   float(candidate_overlap),_json(expected_sources),_json(evidence),float(roi_expansion),roi_output_size)

    @property
    def snapshot_id(self):
        fields={"binding":self.binding.__dict__,"context":self.context_json,
            "rgb":hashlib.sha256(self.native_rgb_bytes).hexdigest(),
            "mask":None if self.mask_bytes is None else hashlib.sha256(self.mask_bytes).hexdigest(),
            "status":self.status,"reason":self.reason,"overlap":self.candidate_overlap,
            "sources":self.sources_json,"quality_evidence":self.quality_evidence_json,
            "roi_expansion":self.roi_expansion,"roi_output_size":self.roi_output_size}
        return hashlib.sha256(_json(fields).encode()).hexdigest()

    def prepare(self,context,rgb,K,*,current_binding,a0_kind,a0_selection_sha256,
                feature_identity_factory,mask_aux_factory):
        _enhanced(a0_kind)
        self.binding.validate()
        _roi_policy(self.roi_expansion,self.roi_output_size)
        if current_binding!=self.binding:raise ValueError("Current track/session/replay/generation differs")
        if a0_selection_sha256!=self.binding.a0_selection_sha256:raise ValueError("Selected A0 differs")
        if not isinstance(context,VideoROIContext):raise ValueError("VideoROIContext required")
        detections=context.consume(rgb,K,b0_sha256=json.loads(self.sources_json)["b0_model_sha256"])
        chosen=[d for d in detections if d["track_id"]==self.binding.track_id]
        if len(chosen)!=1:raise ValueError("Current project track absent")
        now={"frame_id":context.frame_id,"rgb_identity":context.rgb_identity,
            "b0_sha256":context.b0_sha256,"replay_identity":context.replay_identity,
            "K_native":list(context.K_native),"detection":chosen[0]}
        if _json(now)!=self.context_json:raise ValueError("Frame/RGB/K/crop/detection context changed")
        native=np.frombuffer(self.native_rgb_bytes,dtype=np.uint8).reshape(self.native_rgb_shape)
        # Check stored pixels independently, including manually constructed snapshots.
        chw=native.transpose(2,0,1)
        if rgb_digest(chw)!=context.rgb_identity:raise ValueError("Snapshot RGB bytes differ")
        b=self.binding
        image_id=f"video:{b.session_id}:frame:{b.frame_id}:pts:{b.pts}:{b.time_base}"
        frame=ImageFrame(image_id,native,b.pixel_space_id,array_digest(native),"video_predicted_native",array_digest(native))
        bounds=tuple(virtual_roi(chosen[0]["bbox_full_xyxy"],self.roi_expansion))
        roi_id=hashlib.sha256(_json({"snapshot":self.snapshot_id,"bounds":bounds,"size":self.roi_output_size}).encode()).hexdigest()
        identity=feature_identity_factory(image_id,chosen[0]["prediction_id"],roi_id,"action4.roi448.v1")
        native_identity=NativeMaskIdentity(image_id,identity.instance_id,b.pixel_space_id)
        mask=None if self.mask_bytes is None else np.frombuffer(self.mask_bytes,dtype=np.bool_).reshape(native.shape[:2])
        if mask is not None and array_digest(mask)!=b.mask_sha256:raise ValueError("Snapshot mask bytes differ")
        if self.status!="available" and mask is not None and mask.any():raise ValueError("Unavailable foreground")
        predictions={}
        if self.status!="absent":
            native_status=self.status
            sam_binding={"session_id":b.session_id,"obj_id":b.native_obj_id,
                "generation":b.mask_generation,"chosen_mask_id":b.mask_sha256 or "explicit_missing",
                "match_basis":"project_track_generation_and_replay_bound"}
            sources=json.loads(self.sources_json)
            meta=prediction_metadata(frame,native_identity,mask,sources,status=native_status,
                assigned_instance_id=identity.instance_id,candidate_overlap=self.candidate_overlap,
                reason=self.reason,sam_binding=sam_binding)
            predictions[native_key(native_identity)]=NativeSamPrediction(meta,mask)
        inputs=prepare_sam_roi_inputs([frame],[RoiRequest(identity,0,bounds,np.asarray(K),b.pixel_space_id)],
                    predictions,expected_sources=json.loads(self.sources_json),mask_aux_factory=mask_aux_factory)
        return PreparedVideoSam(inputs,_prepared_digest(inputs),self.snapshot_id,b.a0_selection_sha256,self.status,
                                self.reason,self.quality_evidence_json,False)


def _roi_policy(expansion,size):
    if type(expansion) not in (int,float) or not np.isfinite(expansion) or expansion<=0:
        raise ValueError("ROI expansion must be finite and positive")
    if type(size) is not int or size!=448:
        raise ValueError("Existing SAM feature protocol requires ROI output size448")


def _enhanced(kind):
    if kind not in ("joint_pose_sam","joint_pose_sam_corner_candidate"):
        raise ValueError("Baseline cannot silently ignore video SAM; compatible enhanced A0 required")


def _prepared_digest(inputs):
    h=hashlib.sha256()
    arrays=[inputs.rgb,inputs.A448,inputs.A112,inputs.A224,inputs.valid_image_support,
            np.stack(inputs.K112),np.stack(inputs.K224)]
    arrays += [getattr(inputs.mask_aux,k) for k in
               ("visible","distance","available","candidate_overlap","dropout")]
    for value in arrays:
        a=value.detach().cpu().numpy() if hasattr(value,"detach") else np.asarray(value)
        h.update(array_digest(a).encode())
    h.update(_json([x.__dict__ for x in inputs.mask_aux.identities]).encode())
    h.update(_json([inputs.reasons,inputs.source_dependencies]).encode())
    h.update(inputs.mask_aux.protocol_id.encode())
    return h.hexdigest()


@dataclass(frozen=True)
class PreparedVideoSam:
    inputs: object
    prepared_sha256: str
    snapshot_id: str
    a0_selection_sha256: str
    upstream_status: str
    upstream_reason: str | None
    quality_evidence_json: str
    quality_certified: bool

    def forward_kwargs(self,*,rgb,K,a0_kind,a0_selection_sha256):
        """Minimal caller hook for sam_geometry_forward: validates actual ROI/K.

        Returns cloned existing MaskAux and identities only. Caller must use
        these kwargs in the enhanced forward; calling this is not inference.
        """
        import torch
        _enhanced(a0_kind)
        if a0_selection_sha256!=self.a0_selection_sha256:raise ValueError("A0 selection changed")
        if _prepared_digest(self.inputs)!=self.prepared_sha256:raise ValueError("Prepared payload was mutated")
        expected=self.inputs.rgb
        if not isinstance(rgb,torch.Tensor) or rgb.dtype!=expected.dtype or not torch.equal(rgb.detach().cpu(),expected):
            raise ValueError("Actual consumer RGB ROI differs")
        if not isinstance(K,torch.Tensor) or K.dtype!=rgb.dtype or K.device!=rgb.device:
            raise ValueError("Tensor K112 must share consumer RGB dtype/device")
        expected_K=torch.as_tensor(np.stack(self.inputs.K112),dtype=K.dtype,device=K.device)
        if not isinstance(K,torch.Tensor) or not torch.equal(K,expected_K):
            raise ValueError("Actual consumer K112 differs")
        aux=self.inputs.mask_aux
        floating=lambda name:getattr(aux,name).detach().clone().to(device=rgb.device,dtype=rgb.dtype)
        owned=type(aux)(visible=floating("visible"),distance=floating("distance"),
            available=floating("available"),candidate_overlap=floating("candidate_overlap"),
            dropout=aux.dropout.detach().clone().to(rgb.device),identities=aux.identities,
            protocol_id=aux.protocol_id)
        return {"mask_aux":owned,"identities":owned.identities}
