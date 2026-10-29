"""Associate unique current-image marker evidence; crop membership is not ownership."""
import hashlib,json
import numpy as np

def _identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def assign_markers(frame_id,observations,body_masks,expected_marker_by_track,*,minimum_coverage,minimum_margin,
                   duplicate_corner_tolerance_px):
    import cv2
    if (type(frame_id) is not int or frame_id<0 or not 0<minimum_coverage<=1
        or not 0<minimum_margin<=1 or not 0<=duplicate_corner_tolerance_px<=5):
        raise ValueError("Explicit finite ownership policy required")
    shapes={m.shape for m in body_masks.values()}
    if len(shapes)>1 or any(not isinstance(m,np.ndarray) or m.dtype!=np.bool_ or m.ndim!=2 for m in body_masks.values()):
        raise ValueError("Current native boolean body masks must share one pixel domain")
    groups=[]
    for observation in observations:
        mid=observation["marker_id"];quad=np.asarray(observation["corners_native"],float)
        if type(mid) is not int or mid<0 or quad.shape!=(4,2) or not np.isfinite(quad).all() or abs(cv2.contourArea(quad.astype(np.float32)))<4:
            raise ValueError("Valid decoded ID and marker-local corner sequence required")
        same=[]
        for i,g in enumerate(groups):
            if g["marker_id"]==mid and all(np.max(np.linalg.norm(quad-np.asarray(q),axis=1))<=duplicate_corner_tolerance_px for q in g["raw_quads"]):
                same.append(i)
        if len(same)>1:raise ValueError("Nonunique duplicate-evidence grouping")
        if same:
            groups[same[0]]["raw_quads"].append(quad.tolist())
            groups[same[0]]["source_observations"].append(observation)
        else:
            groups.append(dict(marker_id=mid,raw_quads=[quad.tolist()],source_observations=[observation]))
    counts={}
    for g in groups:counts[g["marker_id"]]=counts.get(g["marker_id"],0)+1
    output=[]
    for g in groups:
        # Preserve one actual returned quad; never average two measurements into new corners.
        quad=np.asarray(g["raw_quads"][0])
        entry=dict(frame_id=frame_id,marker_id=g["marker_id"],corners_native=quad.tolist(),
            raw_observations=g["source_observations"],duplicate_crop_observation_count=len(g["raw_quads"])-1,
            status="unassigned",track_id=None,coverage={},reason=None)
        entry["observation_id"]=_identity(dict(frame_id=frame_id,marker_id=g["marker_id"],quad=quad.tolist()))
        if counts[g["marker_id"]]>1:entry["reason"]="same_decoded_ID_at_distinct_locations_ambiguous"
        elif not body_masks:entry["reason"]="current_native_support_missing"
        else:
            h,w=next(iter(body_masks.values())).shape
            l=max(0,int(np.floor(quad[:,0].min())));r=min(w,int(np.ceil(quad[:,0].max()))+1)
            t=max(0,int(np.floor(quad[:,1].min())));b=min(h,int(np.ceil(quad[:,1].max()))+1)
            if r<=l or b<=t:entry["reason"]="marker_outside_current_image"
            else:
                poly=np.zeros((b-t,r-l),np.uint8);cv2.fillConvexPoly(poly,np.rint(quad-[l,t]).astype(np.int32),1);support=poly.astype(bool)
                if not support.any():entry["reason"]="empty_marker_footprint"
                else:
                    entry["coverage"]={tid:float((m[t:b,l:r]&support).sum()/support.sum()) for tid,m in body_masks.items()}
                    ranked=sorted(entry["coverage"].items(),key=lambda x:(-x[1],x[0]))
                    tid,coverage=ranked[0];second=ranked[1][1] if len(ranked)>1 else 0.
                    if coverage<minimum_coverage:entry["reason"]="insufficient_current_body_support"
                    elif coverage-second<minimum_margin:entry["reason"]="mixed_or_duplicate_body_support"
                    elif tid not in expected_marker_by_track or expected_marker_by_track[tid]!=g["marker_id"]:
                        entry["reason"]="decoded_ID_conflicts_with_registered_body"
                    else:entry.update(status="assigned_current_image_evidence",track_id=tid,reason="unique_current_support_and_registered_ID_agree")
        output.append(entry)
    return dict(schema="action4.marker_current_assignment.v1",frame_id=frame_id,observations=output,
        input_observation_count=len(observations),unique_observation_count=len(groups),
        physical_ID_is_provisional=True,metric_pose_GT=False,B0_scores_modified=False)
