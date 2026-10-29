"""Predicted visible-mask features for a separate native region camera; never raw RGB or GT."""
import numpy as np
from data.action4_sam_roi_inputs import array_digest


def region_feature(mask, *, expected_mask_sha256, shape_hw):
    if (not isinstance(mask,np.ndarray) or mask.dtype!=np.bool_ or mask.shape!=tuple(shape_hw)
        or array_digest(mask)!=expected_mask_sha256):
        raise ValueError("Exact current predicted native mask required")
    if not mask.any():raise ValueError("Missing region observation cannot be replaced by a fabricated image")
    # Preserve raw predicted support, including overlaps; no crop-created edges.
    pixels=np.repeat((mask.astype(np.uint8)*255)[:,:,None],3,axis=2)
    return np.ascontiguousarray(pixels),dict(kind="predicted_visible_mask_region_feature",
        source_mask_sha256=expected_mask_sha256,mask_pixels=int(mask.sum()),
        false_colour_levels=[0,255],raw_RGB_modified=False,GT_input=False,
        full_body_extent_verified=False,pixel_space_id="raw_video_native_integer_centers")
