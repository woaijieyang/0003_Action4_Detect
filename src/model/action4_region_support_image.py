"""Explicit per-instance SAM observation image for native region likelihood.

This is a visible-support experiment, not a full-body silhouette or raw RGB.
The texture camera must remain raw RGB. Missing/ambiguous pixels are never
used to invent foreground belonging to another object.
"""
import numpy as np

def region_support_images(masks, *, track_ids, ambiguous_pixels=0, unknown_boxes=None):
    if type(ambiguous_pixels) is not int or ambiguous_pixels<0:raise ValueError("Explicit nonnegative ambiguous pixel count required")
    if ambiguous_pixels:raise ValueError("Ambiguous instance pixels cannot be used as region background")
    if len(set(track_ids))!=len(track_ids) or set(masks)!=set(track_ids):
        raise ValueError("Exact unique instance domain required")
    shape=None;binary={}
    for tid in track_ids:
        m=np.asarray(masks[tid])
        if m.ndim!=2 or m.dtype!=np.uint8 or not np.isin(m,[0,255]).all():raise ValueError("Native unique-owner uint8 mask required")
        if shape is not None and m.shape!=shape:raise ValueError("Mixed native pixel spaces")
        shape=m.shape;binary[tid]=m==255
    occupancy=np.sum(np.stack(list(binary.values())),axis=0)
    if (occupancy>1).any():raise ValueError("Instance masks overlap; reject instead of duplicating evidence")
    if unknown_boxes is not None:
        if set(unknown_boxes)!=set(track_ids):raise ValueError("Same-instance complete predicted box domain required")
        for box in unknown_boxes.values():
            b=np.asarray(box,float)
            if b.shape!=(4,) or not np.isfinite(b).all() or (b[2:]<=b[:2]).any():raise ValueError("Finite positive predicted complete box required")
    images={};audit={}
    for tid,m in binary.items():
        values=np.zeros(shape,np.uint8)
        if unknown_boxes is not None:
            b=np.asarray(unknown_boxes[tid],float);margin=max(2.,.05*float(np.linalg.norm(b[2:]-b[:2])))
            l,t=np.maximum(0,np.ceil(b[:2]-margin).astype(int));r,bot=np.minimum([shape[1],shape[0]],np.ceil(b[2:]+margin).astype(int))
            if r>l and bot>t:values[t:bot,l:r]=128
        values[m]=255
        images[tid]=np.repeat(values[:,:,None],3,axis=2)
        audit[tid]=dict(available=bool(m.any()),foreground_pixels=int(m.sum()),source='current_SAM_unique_owner_visible_support',raw_RGB=False,complete_body_mask=False,GT=False,unknown_pixels=int((values==128).sum()),negative_pixels=int((values==0).sum()),unknown_source=None if unknown_boxes is None else 'predicted_complete_box_plus_max2px_5percent_diagonal_margin',negative_is_measured_background=False)
    return images,audit
