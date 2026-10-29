"""Exhaustive trace audit; numerical/identity validity is not visual accuracy."""
import numpy as np

def audit(rows,expected_frames):
    rows=list(rows);expected_frames=list(expected_frames)
    if len(rows)!=len(expected_frames):raise ValueError("Missing or extra video frames")
    counts={"frames":len(rows),"initializing":0,"two_observed":0,"one_observed":0,
            "zero_observed":0,"lost_instances":0,"predicted_instances":0,
            "shared_mask_rejections":0,"shared_box_rejections":0}
    per_frame=[]
    for row,expected in zip(rows,expected_frames):
        if row["frame"]!=expected:raise ValueError("Video frame/pixel/timestamp identity mismatch")
        fid=expected["global_frame_id"];fused=row["fused"]
        if fused["frame_id"]!=fid or fused["mode"]!="video_inference":
            raise ValueError("Fused frame identity differs")
        tracks=fused["tracks"];seen=set();assigned=set();used_pixels=set();observed=0
        initializing=fused.get("initialization_status")=="unconfirmed"
        if initializing and tracks:raise ValueError("Unconfirmed prefix invented tracks")
        counts["initializing"]+=initializing
        problems=[]
        for track in tracks:
            tid=track["track_id"]
            if tid in seen:raise ValueError("Duplicate track ID")
            seen.add(tid);status=track["status"];box=track["bbox_for_roi"]
            if status not in ("observed","predicted","lost"):raise ValueError("Unknown track status")
            if status=="lost":
                counts["lost_instances"]+=1
                if box is not None:raise ValueError("Lost track still publishes ROI")
                problems.append({"track_id":tid,"reason":"lost"})
            else:
                b=np.asarray(box,float)
                if b.shape!=(4,) or not np.isfinite(b).all() or not (b[2:]>b[:2]).all():
                    raise ValueError("Invalid full-body ROI")
                if status=="predicted":
                    counts["predicted_instances"]+=1
                    problems.append({"track_id":tid,"reason":"prediction_without_current_detection"})
            if status=="observed":
                observed+=1;cid=track["candidate_id"];raw=track["raw_detection"]
                if cid in assigned:raise ValueError("One detection assigned to multiple tracks")
                assigned.add(cid)
                if cid!=raw["candidate_id"] or cid!=f"b0:frame{fid}:grid:{raw['grid_index']}":
                    raise ValueError("Current detection identity mismatch")
            elif track["candidate_id"] is not None or track["raw_detection"] is not None:
                raise ValueError("Missing observation falsely carries a current candidate")
            if track["sam_support"]=="used_for_association_only":
                evidence=row["sam_evidence"][tid];pixel=evidence.get("pixel_sha256")
                if not pixel:raise ValueError("Used mask has no pixel identity")
                if pixel in used_pixels:raise ValueError("Copied mask counted as independent support")
                used_pixels.add(pixel)
        counts["two_observed"]+=observed==2;counts["one_observed"]+=observed==1
        counts["zero_observed"]+=observed==0
        counts["shared_mask_rejections"]+=len(row.get("sam_support_rejections",[]))
        counts["shared_box_rejections"]+=len(fused.get("candidate_rejections",[]))
        per_frame.append({"frame_id":fid,"observed_instances":observed,
                          "initializing":initializing,"problems":problems})
    return {"status":"TRACE_ENGINEERING_VALID_NOT_DETECTION_ACCURACY","counts":counts,
            "per_frame":per_frame,"ground_truth_available":False,
            "two_observed_is_not_two_correct":True}
