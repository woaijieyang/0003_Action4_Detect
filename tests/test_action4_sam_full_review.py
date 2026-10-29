import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from data.action4_paths import PROJECT_ROOT
from model.action4_sam_supported_boxes import SamSupportedBoxes
from eval.action4_sam_full_review import (ReceiptVerifier,ReplayAudit,candidates_for_frame,
    masks_for_frame,validate_completion)

def frame(fid):
    return {"local_index":fid,"global_frame_id":fid,"pts":fid*400,"image":f"frame{fid}.png","sha256":"a"*64}

def native_fixture():
    region={"width":40,"height":30,"time_base":[1,11988],"pixel_space_id":"native"}
    raw={"frame_index":4,"outputs":{"out_obj_ids":np.array([0,1]),"out_binary_masks":np.zeros((2,30,40),bool),
        "out_probs":np.array([.9,.8],np.float32),"out_boxes_xywh":np.zeros((2,4))}}
    raw["outputs"]["out_binary_masks"][0,5:15,5:15]=True
    raw["outputs"]["out_binary_masks"][1,5:15,23:33]=True
    scores={"frame":4,"session_id":"session","available":True,"objects":{"0":{"logit":1.0},"1":{"logit":.8}}}
    status={"frame":4,"global_frame_id":4,"pts":1600,"time_base":[1,11988],"pixel_space_id":"native",
        "RGB_sha256":"a"*64,"session_id":"session","generation":0,"accuracy_accepted":False,
        "native_scores":scores,"history":{"frame":4,"reverse":False,"counts":[15,15]}}
    status["instances"]=[{"obj_id":i,"track_id":f"track_{i:02d}","generation":0,"class_quality_certified":False,
        "public_response_present":True,"mask_pixels":100,"status":"available",
        "public_out_prob":float(raw["outputs"]["out_probs"][i]),"native_tracking_score":scores["objects"][str(i)]} for i in [0,1]]
    return status,raw,frame(4),region,"session"


class FullReview(unittest.TestCase):
    def test_actual_complete_and_close_required(self):
        good={"status":"FULL2850_COVERAGE_JOINT_NATIVE_BASELINE_NOT_ACCURACY","original_frames":2850,"propagated_frames":2846,
            "preseed_unconfirmed":[0,1,2,3],"first_frame":4,"last_frame":2849,
            "joint_obj_map":{"0":"track_00","1":"track_01"},"causal_forward":True,"auto_correction":False,
            "closed":True,"accuracy_accepted":False,"single_session":"session"}
        close={"is_success":True,"gpu_mem":{"active_session_count":0}}
        self.assertEqual(validate_completion(good,close),"session")
        for k,v in [("closed",False),("propagated_frames",2845),("original_frames",2849),("causal_forward",False)]:
            bad=dict(good,**{k:v})
            with self.assertRaises(ValueError):validate_completion(bad,close)
        with self.assertRaises(ValueError):validate_completion(good,{"is_success":False})
    def test_B0_source_frame_and_exact_candidate_ID(self):
        row={"frame":frame(4),"pool":{"schema":"action4.b0_association_pool.v1",
             "association_candidates":[{"grid_index":7,"score":.9,"bbox_full_xyxy":[1,1,9,9]}]}}
        original=copy.deepcopy(row);out=candidates_for_frame(row,frame(4))
        self.assertEqual(out[0]["candidate_id"],"b0:frame4:grid:7");self.assertEqual(row,original)
        for key,value in [("pts",0),("sha256","b"*64),("image","different")]:
            wrong=frame(4);wrong[key]=value
            with self.assertRaises(ValueError):candidates_for_frame(row,wrong)
        row["pool"]["association_candidates"]*=2
        with self.assertRaises(ValueError):candidates_for_frame(row,frame(4))
    def test_native_identity_and_shape_fail_closed(self):
        data=native_fixture();m,u=masks_for_frame(*data);self.assertEqual(u,[])
        for key,value in [("frame",5),("RGB_sha256","b"*64),("pts",0),("session_id","other"),
                          ("generation",1),("pixel_space_id","other")]:
            d=copy.deepcopy(data);d[0][key]=value
            with self.assertRaises(ValueError):masks_for_frame(*d)
        for key,value in [("out_obj_ids",np.array([0,0])),("out_binary_masks",np.zeros((2,1,1),bool)),
                          ("out_probs",np.array([np.nan,.8]))]:
            d=copy.deepcopy(data);d[1]["outputs"][key]=value
            with self.assertRaises(ValueError):masks_for_frame(*d)
    def test_missing_mask_status_not_fabricated(self):
        d=copy.deepcopy(native_fixture())
        for key in d[1]["outputs"]:d[1]["outputs"][key]=d[1]["outputs"][key][:1]
        d[0]["instances"][1].update(status="target_visibility_unknown",mask_pixels=0,public_response_present=False,public_out_prob=None)
        masks,missing=masks_for_frame(*d)
        self.assertIsNone(masks["track_01"]);self.assertEqual(missing,["track_01"])
        d[0]["instances"][1]["status"]="available"
        with self.assertRaises(ValueError):masks_for_frame(*d)
    def test_support_counts_distinct_and_lost_spans(self):
        seeds=[{"track_id":"track_00","bbox_native":[5,5,15,15]},{"track_id":"track_01","bbox_native":[23,5,33,15]}]
        tracker=SamSupportedBoxes(seeds);audit=ReplayAudit(frame_count=12)
        for i in range(4):audit.add({"frame":frame(i),"status":"initialization","fused":{"frame_id":i,"tracks":[]},"native_unavailable":[]})
        masks,_=masks_for_frame(*native_fixture())
        for i in range(4,12):
            available=masks if i==4 else {"track_00":None,"track_01":None}
            fused=tracker.update(i,i/30,[],available)
            audit.add({"frame":frame(i),"status":"replayed","fused":fused,
                "native_unavailable":[] if i==4 else ["track_00","track_01"]})
        summary=audit.finish()
        for counts in summary["counts"].values():
            self.assertEqual(counts,{"observed":0,"sam_supported":1,"predicted":5,"lost":2})
        self.assertEqual(summary["initialization_frames"],[0,1,2,3])
        self.assertFalse(summary["GT_accuracy_measured"]);self.assertFalse(summary["A0_connected"])
        self.assertEqual([(s["start"],s["end"]) for s in summary["lost_spans"]],[(10,11),(10,11)])
    def test_absent_frames_cannot_drop_and_preseed_empty(self):
        a=ReplayAudit(5)
        with self.assertRaises(ValueError):a.add({"frame":frame(1),"fused":{"frame_id":1,"tracks":[]},"status":"initialization","native_unavailable":[]})
        b=ReplayAudit(5)
        b.add({"frame":frame(0),"fused":{"frame_id":0,"tracks":[]},"status":"initialization","native_unavailable":[]})
        with self.assertRaises(ValueError):b.finish()
        c=ReplayAudit(5)
        with self.assertRaises(ValueError):c.add({"frame":frame(0),"fused":{"frame_id":0,"tracks":[{}]},"status":"initialization","native_unavailable":[]})
    def test_shared_candidate_rejected_overlapping_boxes_not_rejected(self):
        a=ReplayAudit(5)
        for i in range(4):a.add({"frame":frame(i),"status":"initialization","fused":{"frame_id":i,"tracks":[]},"native_unavailable":[]})
        tracks=[{"track_id":f"track_{i:02d}","status":"observed","candidate_id":f"b0:frame4:grid:{i}","bbox_for_roi":[1,1,10,10],"sam_current_support":False,"raw_detection":{"candidate_id":f"b0:frame4:grid:{i}","grid_index":i,"bbox_full_xyxy":[1,1,10,10]}} for i in [0,1]]
        a.add({"frame":frame(4),"fused":{"frame_id":4,"tracks":tracks},"native_unavailable":[]});a.finish()
        b=ReplayAudit(5)
        for i in range(4):b.add({"frame":frame(i),"status":"initialization","fused":{"frame_id":i,"tracks":[]},"native_unavailable":[]})
        tracks[1]["candidate_id"]="b0:frame4:grid:0"
        with self.assertRaises(ValueError):b.add({"frame":frame(4),"fused":{"frame_id":4,"tracks":tracks},"native_unavailable":[]})
    def test_tracker_status_contract_invalid_combinations_fail(self):
        def fresh():
            a=ReplayAudit(5)
            for i in range(4):a.add({"frame":frame(i),"status":"initialization","fused":{"frame_id":i,"tracks":[]},"native_unavailable":[]})
            tracks=[{"track_id":f"track_{i:02d}","status":"observed","candidate_id":f"b0:frame4:grid:{i}",
                "bbox_for_roi":[1,1,10,10],"sam_current_support":False,
                "raw_detection":{"candidate_id":f"b0:frame4:grid:{i}","grid_index":i,"bbox_full_xyxy":[1,1,10,10]}} for i in [0,1]]
            return a,{"frame":frame(4),"fused":{"frame_id":4,"tracks":tracks},"native_unavailable":[]}
        changes=[
            {"status":"lost","candidate_id":None,"raw_detection":None},
            {"bbox_for_roi":None},
            {"status":"sam_supported","candidate_id":None,"raw_detection":None,"sam_current_support":False},
            {"raw_detection":None},
            {"raw_detection":{"candidate_id":"b0:frame3:grid:0","grid_index":0,"bbox_full_xyxy":[1,1,10,10]}},
            {"raw_detection":{"candidate_id":"b0:frame4:grid:0","grid_index":9,"bbox_full_xyxy":[1,1,10,10]}},
            {"status":"predicted","candidate_id":None,"raw_detection":None,"sam_current_support":True},
        ]
        for change in changes:
            with self.subTest(change=change):
                a,row=fresh();row["fused"]["tracks"][0].update(change)
                with self.assertRaises(ValueError):a.add(row)
        a,row=fresh();row["native_unavailable"]=["track_00","track_00"]
        with self.assertRaises(ValueError):a.add(row)

    def test_raw_receipt_header_frame_identity_and_shape(self):
        out=PROJECT_ROOT/"outputs/action4_sam_joint_full2850_review_v002";out.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=out) as folder:
            directory=Path(folder);a=np.ones((1,3,4),bool)
            np.savez_compressed(directory/"raw.npz",mask=a)
            header={"frame_index":4,"mask":{"array_key":"mask","shape":[1,3,4],"dtype":"bool"}}
            (directory/"raw.json").write_text(json.dumps(header))
            receipt={"arrays":str(directory/"raw.npz"),"header":str(directory/"raw.json"),
                "sha256":hashlib.sha256((directory/"raw.npz").read_bytes()).hexdigest(),
                "header_sha256":hashlib.sha256((directory/"raw.json").read_bytes()).hexdigest()}
            v=ReceiptVerifier();result=v.raw(receipt,directory)
            self.assertEqual(result["frame_index"],4);np.testing.assert_array_equal(result["mask"],a)
            with self.assertRaises(ValueError):v.raw(receipt,directory/"other")
            header["mask"]["shape"]=[1,4,3];(directory/"raw.json").write_text(json.dumps(header))
            receipt["header_sha256"]=hashlib.sha256((directory/"raw.json").read_bytes()).hexdigest()
            with self.assertRaises(ValueError):ReceiptVerifier().raw(receipt,directory)

    def test_receipt_hash_and_mutation_detection(self):
        out=PROJECT_ROOT/"outputs/action4_sam_joint_full2850_review_v002";out.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=out) as folder:
            p=Path(folder)/"item";p.write_bytes(b"actual")
            sha=hashlib.sha256(p.read_bytes()).hexdigest();v=ReceiptVerifier()
            v.verify(p,sha);v.verify(p,sha)
            self.assertEqual(len(v.verified),1)
            p.write_bytes(b"changed")
            with self.assertRaises(ValueError):v.verify(p,sha)


if __name__=="__main__":unittest.main()
