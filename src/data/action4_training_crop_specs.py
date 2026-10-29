"""Resolved training crops shared by RGB, K, masks and geometry targets.

This packet transports an already chosen crop. It does not select predictions,
verify their scores, create SAM prompts, or authorize training.
"""
from dataclasses import dataclass, asdict
import math

ROUTES = ("gt_jitter", "prediction_full", "prediction_coarse", "extra_offset")

@dataclass(frozen=True)
class TrainingCropSpec:
    image_id: str
    instance_id: str
    box_xyxy: tuple
    requested_route: str
    route: str
    augmentation_seed: int
    policy_sha256: str
    reason: str
    prediction_id: str | None = None

    def __post_init__(self):
        for key in ("image_id","instance_id","reason"):
            if not isinstance(getattr(self,key),str) or not getattr(self,key).strip():
                raise ValueError("Explicit crop identity/reason required")
        if self.requested_route not in ROUTES or self.route not in ROUTES:
            raise ValueError("Unknown crop route")
        if type(self.augmentation_seed) is not int or not 0 <= self.augmentation_seed < 2**64:
            raise ValueError("Explicit deterministic unsigned crop seed required")
        sha=self.policy_sha256
        if not isinstance(sha,str) or len(sha)!=64 or sha=="0"*64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError("Explicit crop policy SHA required")
        if not isinstance(self.box_xyxy,(list,tuple)) or len(self.box_xyxy)!=4:
            raise ValueError("Continuous native XYXY crop box required")
        if any(type(x) not in (int,float) or not math.isfinite(x) for x in self.box_xyxy):
            raise ValueError("Finite crop coordinates required")
        box=tuple(float(x) for x in self.box_xyxy)
        if box[2]<=box[0] or box[3]<=box[1]:
            raise ValueError("Positive crop extent required")
        object.__setattr__(self,"box_xyxy",box)
        if self.route.startswith("prediction_"):
            if not isinstance(self.prediction_id,str) or not self.prediction_id:
                raise ValueError("Predicted crop must name its upstream candidate")
        elif self.prediction_id is not None:
            raise ValueError("GT/offset crop must not masquerade as a predicted candidate")

    def metadata(self):
        return dict(asdict(self),schema="action4.training_crop_spec.v1",
                    pixel_convention="continuous_native_box_boundaries",
                    expansion_applied_by_rgb=2.25,
                    upstream_prediction_verification="caller_responsibility")

def validate_crop_specs(items,specs):
    if not isinstance(specs,(list,tuple)) or len(specs)!=len(items):
        raise ValueError("Exactly one typed crop per selected unit required")
    metadata=[];boxes=[]
    for item,spec in zip(items,specs):
        if not isinstance(spec,TrainingCropSpec):raise ValueError("Typed TrainingCropSpec required")
        record=item["record"];obj=record["instances"][item["instance"]]
        if record.get("split")!="train":raise ValueError("Training crop specs cannot enter evaluation")
        if (spec.image_id,spec.instance_id)!=(record["sample_id"],str(obj["instance_id"])):
            raise ValueError("Crop/image/GT-unit identity mismatch")
        boxes.append(list(spec.box_xyxy));metadata.append(spec.metadata())
    return boxes,metadata
