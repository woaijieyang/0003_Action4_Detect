"""Authenticated current video masks -> real resolved RGB ROI -> SAM-enhanced A0.

No still-image cache receipt is fabricated. No GT, future-frame support or model call.
"""
from dataclasses import dataclass,field
import os,json,hashlib
import numpy as np
from data.action4_sam_roi_inputs import ImageFrame,verify_image_frame,array_digest,SamRoiInputs,_native_nearest_grid,_sample_native_nearest_on_grid,_sample_native_valid_on_grid
from data.action4_dataset import prepare_rois
from data.action4_roi_contract import RoiSpec
from model.action4_sam_feature_adapter import FeatureIdentity,MaskAux

_TOKEN=object()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False)

def validate_bootstrap_frame(record,source,frame_id,b0_sha256):
    if (record.get("frame_id")!=frame_id or record.get("pts")!=source["pts"]
            or record.get("rgb_sha256")!=source["sha256"]
            or record.get("prediction",{}).get("identity",{}).get("b0_sha256")!=b0_sha256):
        raise ValueError("Offline bootstrap current B0/RGB/frame/PTS source differs")

@dataclass(frozen=True)
class VideoSamFeatureContext:
    frame: object
    video_context: object
    K_json: str
    selected_json: str
    receipt_json: str
    masks: tuple
    b0_sha256: str
    sam_model_sha256: str
    source_files: tuple
    provider: object=field(repr=False,compare=False)
    _token: object=field(default=None,init=False,repr=False,compare=False)
    _pid: int=field(default=-1,init=False,repr=False,compare=False)

    def __reduce_ex__(self,protocol):raise TypeError("Video SAM context must be verified in consumer process")

    @property
    def expected_sources(self):
        return dict(schema="action4.current_video_sam_feature_sources.v1",b0_model_sha256=self.b0_sha256,
            sam_model_sha256=self.sam_model_sha256,replay_identity=self.video_context.replay_identity)

    @classmethod
    def load(cls,provider,frame_id,K,*,expected_b0_sha256,expected_sam_model_sha256,bootstrap=None):
        from eval.action4_reanchor_roi_provider import ReanchorROIProvider
        if not isinstance(provider,ReanchorROIProvider) or provider.b0_sha!=expected_b0_sha256:
            raise ValueError("Authenticated current video provider and fixed B0 required")
        rgb,context=provider.load(frame_id,K)
        bootstrap_files=()
        if bootstrap is not None:
            # Preserve the existing explicitly offline frame-4 confirmation policy.
            # It supplies current B0 boxes for frames 0..3, never future RGB/masks.
            if frame_id>=4 or set(bootstrap)!={"raw_B0_operating_reference","bootstrap_seeds"}:
                raise ValueError("Offline bootstrap is restricted to the registered preseed frames")
            import itertools
            v=provider.verifier
            raw_ref=bootstrap["raw_B0_operating_reference"];seed_ref=bootstrap["bootstrap_seeds"]
            raw_path=v.verify(raw_ref["path"],raw_ref["sha256"])
            seed_path=v.verify(seed_ref["path"],seed_ref["sha256"])
            with raw_path.open() as stream:first=[json.loads(line) for line in itertools.islice(stream,4)]
            seeds=json.loads(seed_path.read_text())["seeds"]
            if len(first)!=4:raise ValueError("Incomplete offline bootstrap frame domain")
            validate_bootstrap_frame(first[frame_id],provider.rows[frame_id]["frame"],frame_id,expected_b0_sha256)
            identity=hashlib.sha256(json.dumps({"B0":raw_ref,"seeds":seed_ref,
                "policy":"shared_existing_offline_confirmation_at4"},sort_keys=True).encode()).hexdigest()
            from eval.action4_roi_video import bootstrap_context
            context=bootstrap_context(provider,frame_id,rgb,K,first[frame_id],seeds,identity)
            bootstrap_files=((str(raw_path),raw_ref["sha256"]),(str(seed_path),seed_ref["sha256"]))
        selected=context.select_for_a0(context.consume(rgb,K,b0_sha256=expected_b0_sha256))
        masks,receipt=provider.current_native_masks(frame_id)
        source=provider.rows[frame_id]["frame"]
        if receipt["RGB_sha256"]!=source["sha256"] or receipt["pts"]!=source["pts"]:
            raise ValueError("Video mask/RGB/PTS source differs")
        owners=[o for o in provider.owners if o["start"]<=frame_id]
        owner=owners[-1] if owners else provider.owners[0]  # preseed RGB fallback, no mask assumed
        root=owner["native_root"];v=provider.verifier
        manifest=json.loads(v.verify(root/"source/SNAPSHOT_MANIFEST.json",owner["source_sha"]).read_text())
        cfg=json.loads(v.verify(root/"source/config.json",manifest["files_sha256"]["config.json"]).read_text())
        from data.action4_paths import PROJECT_ROOT,contained_path
        asset_path=contained_path(PROJECT_ROOT,cfg["native"]["asset_receipt"],exists=True)
        if asset_path!=(root/"source/inputs/ASSET_VALIDATION.json").resolve():
            raise ValueError("Video SAM asset receipt path differs")
        asset=json.loads(v.verify(asset_path,manifest["files_sha256"]["inputs/ASSET_VALIDATION.json"]).read_text())
        if asset["sha256"]!=expected_sam_model_sha256:raise ValueError("Actual video SAM weights differ from enhanced training recipe")
        pixels=np.moveaxis(rgb.detach().cpu().numpy(),0,-1)
        frame=verify_image_frame(ImageFrame("video:"+context.replay_identity+":frame:"+str(frame_id),
            pixels,receipt["pixel_space_id"],source["sha256"],"video_native_original",array_digest(pixels)))
        packed=[];seen_masks=set()
        for track,mask in sorted(masks.items()):
            if mask.dtype!=np.bool_ or mask.shape!=pixels.shape[:2] or array_digest(mask)!=receipt["mask_sha256"][track]:
                raise ValueError("Current native body mask malformed")
            digest=array_digest(mask)
            if digest in seen_masks:raise ValueError("One identical current mask cannot support two body feature inputs")
            seen_masks.add(digest)
            packed.append((track,mask.shape,mask.tobytes()))
        files=((str(root/"source/SNAPSHOT_MANIFEST.json"),owner["source_sha"]),
            (str(root/"source/config.json"),manifest["files_sha256"]["config.json"]),
            (str(asset_path),manifest["files_sha256"]["inputs/ASSET_VALIDATION.json"]),
            (source["image"],source["sha256"]))+bootstrap_files
        result=cls(frame,context,canonical(np.asarray(K,float).tolist()),canonical(selected),canonical(receipt),
            tuple(packed),expected_b0_sha256,expected_sam_model_sha256,files,provider)
        object.__setattr__(result,"_token",_TOKEN);object.__setattr__(result,"_pid",os.getpid())
        return rgb,context,result

    def consume(self,rgb,K,context,detections,*,b0_sha256):
        if self._token is not _TOKEN or self._pid!=os.getpid():raise ValueError("Verified current video feature context required")
        if context is not self.video_context or b0_sha256!=self.b0_sha256 or canonical(np.asarray(K,float).tolist())!=self.K_json:
            raise ValueError("Video frame context/model/camera changed")
        pixels=np.moveaxis(rgb.detach().cpu().numpy() if hasattr(rgb,"detach") else np.asarray(rgb),0,-1)
        if not np.array_equal(pixels,self.frame.pixels):raise ValueError("Video feature RGB changed")
        actual=context.select_for_a0(context.consume(rgb,K,b0_sha256=b0_sha256))
        if canonical(actual)!=self.selected_json or canonical(detections)!=self.selected_json:
            raise ValueError("Actual resolved ROI/instance selection differs from mask context")
        for path,sha in self.source_files:self.provider.verifier.verify(path,sha)
        _,receipt=self.provider.current_native_masks(context.frame_id)
        if canonical(receipt)!=self.receipt_json:raise ValueError("Current native response identity changed")
        return {track:np.frombuffer(data,dtype=np.bool_).reshape(shape) for track,shape,data in self.masks},json.loads(self.receipt_json)


def prepare_video_pipeline_sam(rgb,K,detections,context,*,video_context,b0_sha256,roi_expansion):
    import torch
    from scipy.ndimage import distance_transform_edt
    masks,receipt=context.consume(rgb,K,video_context,detections,b0_sha256=b0_sha256)
    boxes=[d["bbox_for_a0"] for d in detections]
    rois,A448,specs=prepare_rois([rgb],boxes,[0]*len(boxes),roi_size=448,expansion=roi_expansion,device="cpu",return_specs=True)
    A448=np.asarray(A448);A112=np.stack([s.affine(112) for s in specs]);A224=np.stack([s.affine(224) for s in specs])
    n=len(specs);visible=torch.zeros(n,1,448,448);distance=torch.zeros_like(visible);valid=torch.zeros_like(visible)
    available=torch.zeros(n,1,1,1);overlap=torch.zeros(n,1);dropout=torch.zeros(n,dtype=torch.bool)
    identities=[];dependencies=[];reasons=[]
    for i,(d,spec) in enumerate(zip(detections,specs)):
        pid=d["prediction_id"]
        if pid!=f"video:{d['track_id']}:frame:{video_context.frame_id}":raise ValueError("Current video instance ID differs")
        rid=hashlib.sha256(canonical(dict(image=context.frame.image_id,prediction=pid,bounds=spec.continuous_bounds,size=448)).encode()).hexdigest()
        identities.append(FeatureIdentity(context.frame.image_id,pid,rid,"action4.roi448.v1"))
        shape=context.frame.pixels.shape[:2];grid=_native_nearest_grid(shape,A448[i])
        valid[i]=_sample_native_valid_on_grid(shape,grid)
        mask=masks.get(d["track_id"]);reason="current_video_mask_unavailable"
        if mask is not None:
            yy,xx=np.nonzero(mask)
            overlap_box=d.get("bbox_b0_raw")
            if overlap_box is None:overlap_box=d["bbox_for_a0"]
            box=np.asarray(overlap_box,float)
            overlap[i,0]=float(((xx>=box[0])&(xx<box[2])&(yy>=box[1])&(yy<box[3])).mean()) if len(xx) else 0.
            sampled=_sample_native_nearest_on_grid(mask,grid);m=sampled[0].numpy().astype(bool)
            if m.any():
                visible[i]=sampled;distance[i,0]=torch.from_numpy(np.clip((distance_transform_edt(m)-distance_transform_edt(~m))/16,-1,1).astype(np.float32))
                available[i]=1;reason="current_video_native_mask"
            else:reason="current_video_mask_empty_after_resolved_ROI"
        reasons.append(reason)
        dependencies.append(dict(protocol="action4.video_sam_features.v1",sources=context.expected_sources,
            current_native_receipt=receipt,track_id=d["track_id"],frame_id=video_context.frame_id,
            roi_decision_id=d["roi_decision_id"],bbox_for_a0=list(d["bbox_for_a0"]),roi_spec=spec.record(),
            input_source="current_video_predicted_visible_mask_not_GT",still_image_cache_used=False,
            availability_reason=reason,candidate_overlap_semantics=("fraction_of_current_mask_in_raw_B0_box" if d.get("bbox_b0_raw") is not None
                else "fraction_of_current_mask_in_estimated_resolved_box_no_current_B0"),
            use_as_training_negative=False))
    aux=MaskAux(visible,distance,available,overlap,dropout,tuple(identities))
    return SamRoiInputs(rois,aux,A448,A112,A224,valid,tuple(reasons),tuple(dependencies),
        tuple(a@K for a in A112),tuple(a@K for a in A224)),specs
