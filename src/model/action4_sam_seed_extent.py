"""Whole-instance PVS seed prompts and predicted-reference extent checks.

This is a versioned box-corner candidate API, not the old 0/1 PromptEvent.
The reference is frozen predicted discovery, never certified segmentation GT.
"""
import math
import numpy as np

PROTOCOL="action4.sam.pvs_box_corners.v1"

def box_corner_prompt(session_id,frame_index,obj_id,box,width,height):
    if not isinstance(session_id,str) or not session_id:raise ValueError("Session identity required")
    if any(type(v) is not int or v<0 for v in (frame_index,obj_id)):raise ValueError("Local frame/object integer required")
    if any(type(v) is not int or v<=0 for v in (width,height)):raise ValueError("Native image dimensions required")
    b=np.asarray(box,dtype=float)
    if b.shape!=(4,) or not np.isfinite(b).all() or not (0<=b[0]<b[2]<=width and 0<=b[1]<b[3]<=height):
        raise ValueError("Explicit in-image native support box required; no implicit clipping")
    return {"type":"add_prompt","session_id":session_id,"frame_index":frame_index,"obj_id":obj_id,
            "points":[[float(b[0]/width),float(b[1]/height)],[float(b[2]/width),float(b[3]/height)]],
            "point_labels":[2,3],"rel_coordinates":True,"clear_old_points":True}

def discovery_box(mask):
    _mask(mask);ys,xs=np.nonzero(mask)
    if not len(xs):raise ValueError("Nonempty discovery support required")
    return [int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]

def _mask(mask):
    if not isinstance(mask,np.ndarray) or mask.dtype!=np.bool_ or mask.ndim!=2 or min(mask.shape)<=0:
        raise ValueError("Native nonempty-space bool mask required")

def extent_gate(reference,current,*,minimum_coverage=.8,minimum_purity=.8):
    _mask(reference);_mask(current)
    if reference.shape!=current.shape:raise ValueError("Native pixel spaces differ")
    if any(not math.isfinite(v) or not 0<v<=1 for v in (minimum_coverage,minimum_purity)):
        raise ValueError("Explicit coverage and purity thresholds required")
    a=int(reference.sum());b=int(current.sum())
    if not a:raise ValueError("Missing discovery reference cannot certify a seed")
    overlap=int(np.count_nonzero(reference&current));coverage=overlap/a;purity=overlap/b if b else 0.
    return {"protocol":PROTOCOL,"discovery_pixels":a,"seed_pixels":b,"intersection_pixels":overlap,
            "discovery_coverage":coverage,"seed_purity":purity,"mask_IoU":overlap/(a+b-overlap),
            "area_ratio":b/a,"minimum_coverage":minimum_coverage,"minimum_purity":minimum_purity,
            "extent_preserved":bool(coverage>=minimum_coverage and purity>=minimum_purity),
            "reference_kind":"frozen_predicted_discovery_mask","GT_used":False,"identity_accuracy_certified":False}


def b0_body_scope_prompt(session_id,frame_index,obj_id,detection,calibration,width,height):
    """Candidate PVS instance scope, derived solely from an admitted B0 box.

    Caller authenticates frame/track/calibration provenance before this call.
    This does not certify the detector's box or schedule a native correction.
    Unlike PCS semantic prompts, native point labels2/3 preserve instance IDs.
    """
    if not isinstance(detection,dict) or not isinstance(calibration,dict):
        raise ValueError("B0 detection and independent calibration receipt required")
    cid=detection.get("candidate_id")
    if not isinstance(cid,str) or not cid or calibration.get("candidate_id")!=cid:
        raise ValueError("Calibration/detection identity mismatch")
    if calibration.get("accepted") is not True:
        raise ValueError("Previously accepted B0 calibration required")
    for key in ("score","refined_quality_score"):
        value=detection.get(key)
        if (isinstance(value,bool) or not isinstance(value,(int,float,np.floating))
                or not math.isfinite(float(value)) or not .9<=value<=1
                or calibration.get(key)!=value):
            raise ValueError("Matching finite high-score B0 calibration required")
    b=np.asarray(detection.get("bbox_full_xyxy"),dtype=float)
    cb=np.asarray(calibration.get("raw_B0_box"),dtype=float)
    if b.shape!=(4,) or cb.shape!=(4,) or not np.array_equal(b,cb):
        raise ValueError("Calibration must bind the original B0 body box")
    request=box_corner_prompt(session_id,frame_index,obj_id,b,width,height)
    x0,y0,x1,y1=b;w=x1-x0;h=y1-y0;cx=(x0+x1)/2;cy=(y0+y1)/2
    points=[(cx,cy)]
    possible=[(x0-.15*w,cy),(x1+.15*w,cy),(cx,y0-.15*h),(cx,y1+.15*h)]
    negatives=[];dropped=0
    for x,y in possible:
        x=float(np.clip(x,0,width-1));y=float(np.clip(y,0,height-1))
        # Clamping at an image edge must not invent a negative inside the body.
        if x0<=x<=x1 and y0<=y<=y1:
            dropped+=1
        elif (x,y) not in negatives:
            negatives.append((x,y))
        else:dropped+=1
    if not negatives:raise ValueError("No legal outside-box negative support")
    points.extend(negatives)
    request["points"].extend([[float(x/width),float(y/height)] for x,y in points])
    request["point_labels"].extend([1]+[0]*len(negatives))
    return request,{"protocol":"action4.sam.pvs_b0_body_scope.v1",
        "candidate_id":cid,"raw_B0_box":b.tolist(),
        "B0_score":float(detection["score"]),
        "refined_quality_score":float(detection["refined_quality_score"]),
        "negative_offset_box_fraction":.15,"dropped_negative_points":dropped,
        "negative_role":"predicted_outside_body_not_certified_background",
        "manual_pixel_inputs":False,"body_scope_certified":False,
        "automatic_correction_scheduler_implemented":False}
