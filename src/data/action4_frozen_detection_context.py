"""Immutable image-only B0 replay. No GT or temporal recovery is accepted."""
from dataclasses import dataclass
import json
import numpy as np
from .action4_sam_roi_inputs import verify_image_frame,array_digest
from .action4_sam_cache_provider import native_rgb

@dataclass(frozen=True)
class FrozenDetectionContext:
    image_id:str
    pixel_sha256:str
    K_native:tuple
    b0_sha256:str
    scoring_policy:str
    detections_json:str
    source_cache_identity:str

    @classmethod
    def from_frame(cls,frame,K,rows,*,b0_sha256,scoring_policy,source_cache_identity):
        f=verify_image_frame(frame)
        ids=[]
        for p in rows:
            if any(k in p for k in ("track_id","temporal_status","bbox_for_a0")):raise ValueError("Image eval cannot consume tracking decisions")
            b=np.asarray(p["bbox_full_xyxy"],float)
            if b.shape!=(4,) or not np.isfinite(b).all() or not (b[2:]>b[:2]).all() or type(p["grid_index"]) is not int:
                raise ValueError("Invalid frozen B0 candidate")
            if not np.isfinite(p["score"]) or not 0<=p["score"]<=1:raise ValueError("Invalid frozen score")
            ids.append(p["grid_index"])
        if len(ids)!=len(set(ids)):raise ValueError("Duplicate cached B0 candidate")
        return cls(f.image_id,f.variant_sha256,tuple(np.asarray(K,float).reshape(-1)),b0_sha256,scoring_policy,
            json.dumps(rows,allow_nan=False),source_cache_identity)
    @classmethod
    def from_sam_frame(cls,frame,K,rows,*,sam_predictions,sam_sources,b0_sha256,scoring_policy,source_cache_identity):
        """Bind both frozen image branches to the identical frame and stable IDs."""
        from .action4_sam_roi_inputs import NativeMaskIdentity,native_key
        f=verify_image_frame(frame)
        if sam_sources.get("b0_model_sha256")!=b0_sha256 or f.variant_id!="native_original":
            raise ValueError("Frozen SAM/B0 original-image source differs")
        for row in rows:
            pid=f"b0:{f.image_id}:grid:{row['grid_index']}"
            if row.get("prediction_id",pid)!=pid:raise ValueError("Frozen detector candidate ID differs from its image/grid")
            if row["score"]>=.3 and native_key(NativeMaskIdentity(f.image_id,pid,f.pixel_space_id)) not in sam_predictions:
                raise ValueError("Operating B0 candidate absent from declared SAM cache domain; missing masks need explicit status records")
        return cls.from_frame(f,K,rows,b0_sha256=b0_sha256,scoring_policy=scoring_policy,source_cache_identity=source_cache_identity)

    def consume(self,rgb,K,*,b0_sha256,scoring_policy):
        if (array_digest(native_rgb(rgb))!=self.pixel_sha256 or b0_sha256!=self.b0_sha256
                or scoring_policy!=self.scoring_policy or tuple(np.asarray(K,float).reshape(-1))!=self.K_native):
            raise ValueError("Frozen detection RGB/K/model/policy differs")
        return json.loads(self.detections_json)
