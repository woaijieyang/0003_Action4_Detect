"""Regression: demo-envelope proxy keeps misses, permits instance-order changes, never scores IDs."""
import copy
import numpy as np
from eval.action4_pose_reference_metrics import compare_pose_envelopes

def main():
    boxes=[[10,20,60,60],[100,20,150,60]]
    reference={"rows":[{"frame_id":0,"reference_candidate":True,"components":[{"bbox_demo_px":b} for b in boxes]},
                       {"frame_id":1,"reference_candidate":False,"components":[]}]}
    scale=np.array([3248/950,2464/720])
    instances=[]
    for b in boxes:
        corners=np.array([[b[0],b[1]],[b[2],b[1]],[b[2],b[3]],[b[0],b[3]]]*2)*scale
        instances.append({"bbox_full_xyxy":(np.array(b)*np.tile(scale,2)).tolist(),"pose_valid":True,"corners_px":corners.tolist()})
    predictions=[{"frame_id":0,"prediction":{"instances":instances[::-1]}},{"frame_id":1,"prediction":{"instances":[]}}]
    result=compare_pose_envelopes(reference,predictions)
    assert result["both_pose_envelope_frames"]==1 and result["unscorable_reference_frames"]==1
    bad=copy.deepcopy(predictions);bad[0]["prediction"]["instances"][0]["pose_valid"]=False
    assert compare_pose_envelopes(reference,bad)["both_pose_envelope_frames"]==0
    bad[0]["prediction"]["instances"].pop()
    assert compare_pose_envelopes(reference,bad)["eligible_frames"]==1
    assert compare_pose_envelopes(reference,bad)["both_box_frames"]==0
    # An ID permutation preserves envelopes: this metric cannot claim corner-ID correctness.
    swapped=copy.deepcopy(predictions)
    for instance in swapped[0]["prediction"]["instances"]:instance["corners_px"]=instance["corners_px"][::-1]
    assert compare_pose_envelopes(reference,swapped)["both_pose_envelope_frames"]==1
    try:compare_pose_envelopes(reference,predictions[:1])
    except ValueError:pass
    else:raise AssertionError("Missing source frames silently accepted")
    print("PASS: one-to-one instance matching, misses/invalid failures, coverage, explicit no-ID scope")
if __name__=="__main__":main()
