"""Development-only reliability gate for native RGB region observations; no GT."""
from dataclasses import dataclass
from model.action4_native_recovery import box_iou

@dataclass
class RegionObservationGate:
    track_id: str
    enabled: bool = True
    last_frame: int = -1

    def observe(self,*,track_id,frame_id,native_box,b0_box,current_B0,current_SAM,prior_available,
                resolved_box=None,resolved_current_support=False):
        if track_id!=self.track_id or type(frame_id) is not int or frame_id<=self.last_frame:
            raise ValueError("Current ordered observation from the same body required")
        self.last_frame=frame_id
        if type(resolved_current_support) is not bool:raise ValueError("Explicit resolved ROI observation state required")
        if resolved_current_support:
            if not (current_B0 is True or current_SAM is True) or resolved_box is None:
                raise ValueError("Resolved ROI needs current B0 or SAM evidence; historical box alone is insufficient")
            box_iou(resolved_box,resolved_box)
            reference=resolved_box;reference_kind="resolved_full_body_ROI_prediction"
            supported=prior_available is True
        else:
            reference=b0_box;reference_kind="raw_B0_legacy_reference"
            supported=current_B0 is True and current_SAM is True and prior_available is True
        overlap=box_iou(native_box,reference) if supported and native_box is not None and reference is not None else None
        if supported and (overlap is None or overlap<.3):
            self.enabled=False;reason="current_resolved_ROI_contradicts_native_region" if resolved_current_support else "current_B0_SAM_contradicts_native_region"
        elif supported and overlap>=.5:
            self.enabled=True;reason="current_evidence_realigns_region"
        elif not supported:
            reason="no_new_region_reliability_evidence_keep_previous_state"
        else:
            reason="hysteresis_keep_previous_state"
        return dict(track_id=track_id,frame_id=frame_id,enabled=self.enabled,reason=reason,
            native_B0_box_IoU=overlap if not resolved_current_support else None,
            native_reference_box_IoU=overlap,reference_kind=reference_kind,
            reference_box=reference,reference_is_geometry_GT=False,disable_below=.3,reenable_at=.5,
            effect="native_n_lines_max_zero_when_disabled",GT_used=False,
            inferred_observation_reliability_not_calibrated_probability=True)
