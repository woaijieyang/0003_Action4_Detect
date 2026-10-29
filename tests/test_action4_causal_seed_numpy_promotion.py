"""Exact producer promotion regression; never tolerance-based native binding."""
import copy,json
from pathlib import Path
import unittest
import numpy as np

from tests import test_action4_clip_replay as fixtures
from eval.action4_sam_b0_sequence import load_instance_seed_points
from eval.action4_sam_image_eval import _box_iou


class CausalSeedPromotionTests(unittest.TestCase):
    def fixture(self):
        owner=fixtures.ClipReplayTests(methodName="runTest");self.addCleanup(owner.doCleanups)
        folder,out,cfg,seeds,refs=owner.point_seed_source()
        # Explicit synthetic producer-v1 receipt, with real source identity and
        # exact raw float32 boxes. Raw bytes are never changed by this fixture.
        prior=json.loads((fixtures.ROOT/refs["config"]["path"]).read_text())
        source=json.loads((fixtures.ROOT/prior["source_snapshot"]/"SNAPSHOT_MANIFEST.json").read_text())
        load_path=fixtures.ROOT/refs["native_load"]["path"];load=json.loads(load_path.read_text())
        load.update(numpy="1.26.4",source_snapshot_identity=source["identity"])
        refs["native_load"]=owner.write(load_path,load)
        binding_path=fixtures.ROOT/refs["binding"]["path"];binding=json.loads(binding_path.read_text())
        receipt=json.loads((fixtures.ROOT/refs["raw_receipt"]["path"]).read_text())
        raw_path=fixtures.ROOT/receipt["arrays"]
        before=owner.ref(raw_path)
        with np.load(raw_path,allow_pickle=False) as z:boxes=z["raw__outputs__out_boxes_xywh"].copy()
        for row,box in zip(binding["native_candidates"],boxes):
            x,y,w,h=box
            native=np.array([float(x)*40,float(y)*40,float(np.float32(x+w))*40,float(np.float32(y+h))*40])
            row["bbox_native_xyxy"]=native.tolist()
            row["bbox_iou_prompt"]=_box_iou(native,np.asarray(seeds[0]["bbox_native"]))
        refs["binding"]=owner.write(binding_path,binding)
        return owner,cfg,seeds,refs,binding_path,binding,load_path,load,raw_path,before

    def test_bound_numpy1_producer_exactly_validates_on_numpy2_without_raw_changes(self):
        owner,cfg,seeds,refs,_,_,_,_,raw,before=self.fixture()
        points,labels,_=load_instance_seed_points(cfg,seeds[0],seeds)
        self.assertEqual(labels,[1,1,0,0])
        self.assertEqual(len(points),4)
        self.assertEqual(owner.ref(raw),before)

    def test_one_ulp_bbox_or_iou_and_other_binding_field_changes_are_rejected(self):
        for field in ("bbox_native_xyxy","bbox_iou_prompt","target_mask_coverage"):
            owner,cfg,seeds,refs,path,binding,_,_,_,_=self.fixture()
            row=binding["native_candidates"][0]
            if field=="bbox_native_xyxy":row[field][0]=float(np.nextafter(row[field][0],np.inf))
            elif field=="bbox_iou_prompt":row[field]=float(np.nextafter(row[field],np.inf))
            else:row[field]-=.001
            refs["binding"]=owner.write(path,binding)
            with self.assertRaisesRegex(ValueError,"native seed binding differs"):
                load_instance_seed_points(cfg,seeds[0],seeds)

    def test_missing_or_wrong_producer_runtime_proof_does_not_guess(self):
        # Current test runtime is NumPy2. Synthetic v1 diagnostics must not
        # receive compatibility replay without a correctly bound v1 identity.
        if np.__version__.split(".",1)[0]!="2":
            self.skipTest("Cross-major refusal is exercised by the NumPy2 consumer suite")
        for kind in ("missing_version","wrong_version","wrong_source"):
            owner,cfg,seeds,refs,_,_,path,load,_,_=self.fixture()
            if kind=="missing_version":load.pop("numpy")
            elif kind=="wrong_version":load["numpy"]="2.5.2"
            else:load["source_snapshot_identity"]="f"*64
            refs["native_load"]=owner.write(path,load)
            with self.assertRaisesRegex(ValueError,"native seed binding differs"):
                load_instance_seed_points(cfg,seeds[0],seeds)



    def replay_fixture(self):
        from eval.action4_clip_replay import ClipReplay
        owner=fixtures.ClipReplayTests(methodName="runTest");self.addCleanup(owner.doCleanups)
        folder,out,args,_=owner.fixture(forward=True)
        cfg=json.loads((fixtures.ROOT/args["config_ref"]["path"]).read_text())
        source=json.loads((fixtures.ROOT/cfg["source_snapshot"]/"SNAPSHOT_MANIFEST.json").read_text())
        seeds=json.loads((out/"SEEDS.json").read_text())["seeds"]
        load_ref=owner.write(out/"NATIVE_LOAD.json",{"status":"LOADED_ZERO_FINAL_MISSING_UNEXPECTED",
            "numpy":"1.26.4","source_snapshot_identity":source["identity"],"fixture":"synthetic diagnostics"})
        args["evidence_refs"][load_ref["path"]]=load_ref["sha256"]
        for seed in seeds:
            folder=out/seed["track_id"];binding_path=folder/"SEED_BINDING.json"
            binding=json.loads(binding_path.read_text());receipt=json.loads((folder/"RAW_RECEIPT.json").read_text())
            with np.load(fixtures.ROOT/receipt["arrays"],allow_pickle=False) as z:
                boxes=z["raw__outputs__out_boxes_xywh"].copy()
            for row,box in zip(binding["native_candidates"],boxes):
                x,y,w,h=box
                native=np.array([float(x)*40,float(y)*40,float(np.float32(x+w))*40,float(np.float32(y+h))*40])
                row["bbox_native_xyxy"]=native.tolist()
                row["bbox_iou_prompt"]=_box_iou(native,np.asarray(seed["bbox_native"]))
            ref=owner.write(binding_path,binding);args["evidence_refs"][ref["path"]]=ref["sha256"]
        return owner,out,args

    def test_clip_replay_reuses_exact_producer_diagnostics(self):
        from eval.action4_clip_replay import ClipReplay
        _,_,args=self.replay_fixture()
        replay=ClipReplay(**args)
        self.assertEqual(len(replay.core_rows()),4)
        self.assertEqual(len(replay.owners),2)
        self.assertEqual(replay.status,"native_complete_not_accuracy")

    def test_clip_replay_still_rejects_one_ulp_and_support_field_tampering(self):
        from eval.action4_clip_replay import ClipReplay
        for field in ("bbox","coverage"):
            owner,out,args=self.replay_fixture()
            path=out/"track_00/SEED_BINDING.json";binding=json.loads(path.read_text())
            row=binding["native_candidates"][0]
            if field=="bbox":row["bbox_native_xyxy"][0]=float(np.nextafter(row["bbox_native_xyxy"][0],np.inf))
            else:row["target_mask_coverage"]-=.001
            ref=owner.write(path,binding);args["evidence_refs"][ref["path"]]=ref["sha256"]
            with self.assertRaisesRegex(ValueError,"Actual unique seed/native object binding differs"):
                ClipReplay(**args)


if __name__=="__main__":
    unittest.main()
