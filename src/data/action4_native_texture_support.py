"""Current visible instance support for native texture observations, not full-body geometry."""
import numpy as np
from data.action4_sam_roi_inputs import array_digest


def prepare_texture_support(masks, receipt, *, frame_id, pts, time_base, rgb_sha256, image_shape, tracks):
    if (receipt.get("frame_id")!=frame_id or receipt.get("pts")!=pts or receipt.get("time_base")!=list(time_base)
        or receipt.get("RGB_sha256")!=rgb_sha256 or receipt.get("pixel_space_id")!="raw_video_native_integer_centers"
        or receipt.get("status")!="current_native_response" or type(receipt.get("generation")) is not int):
        raise ValueError("Current native mask frame/PTS/RGB/generation required")
    if len(set(tracks))!=len(tracks) or not set(masks)<=set(tracks):
        raise ValueError("Explicit unique body IDs required")
    checked={};digests=set()
    for track,mask in masks.items():
        if (not isinstance(mask,np.ndarray) or mask.dtype!=np.bool_ or mask.shape!=tuple(image_shape)
            or receipt.get("per_track_status",{}).get(track)!="current_mask_present"):
            raise ValueError("Current same-native-image bool support required")
        digest=array_digest(mask)
        if receipt["mask_sha256"].get(track)!=digest:raise ValueError("Native support bytes/identity changed")
        if digest in digests:raise ValueError("One mask cannot be duplicated as independent bodies")
        digests.add(digest);checked[track]=mask
    occupancy=np.zeros(image_shape,np.uint8)
    for mask in checked.values():occupancy+=mask
    result={};audit={}
    for track in tracks:
        raw=checked.get(track)
        # Shared pixels have ambiguous ownership: exclude observations, not instances.
        support=np.zeros(image_shape,bool) if raw is None else raw & (occupancy==1)
        result[track]=np.ascontiguousarray(support.astype(np.uint8)*255)
        audit[track]=dict(frame_id=frame_id,track_id=track,generation=receipt["generation"],
            native_mask_sha256=None if raw is None else receipt["mask_sha256"][track],
            support_sha256=array_digest(result[track]),raw_pixels=0 if raw is None else int(raw.sum()),
            texture_support_pixels=int(support.sum()),ambiguous_pixels_excluded=0 if raw is None else int((raw & (occupancy>1)).sum()),
            status="current_support" if support.any() else "no_texture_observation",
            inference_geometry="visible support only; not amodal box or face correspondence")
    return result,audit
