"""CPU contract tests; tiny synthetic images/fake predictor are not dev results."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np
from PIL import Image
from eval import action4_full_zeroshot_eval as m

ROOT=Path("/data/zy/vln/a-test")


class FullDevRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=m.load_config(m.DEFAULT_CONFIG)
        m.activate_snapshot(cls.cfg)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="ENGINEERING_FIXTURE_",dir=m.output_root()/"tmp")
        self.folder=Path(self.temp.name)
        self.image=self.folder/"eval"/"rgb.png";self.image.parent.mkdir()
        Image.fromarray(np.zeros((16,16,3),np.uint8)).save(self.image)
        self.profile_path=self.folder/"fixture_profile.npz";self.profile_path.write_bytes(b"FAKE_UNIT_PROFILE_NO_MODEL")
        self.profile={"name":"front","path":str(self.profile_path.relative_to(ROOT)),
                      "sha256":m.sha(self.profile_path),"reference_id":"fixture_ref","text":False}
        self.row={"index":0,"sample_id":"ENGINEERING_FIXTURE_IMAGE","image":m.checked_file(self.image,m.sha(self.image)),
                  "native_shape":[16,16],"original_gt_count":2}
        base={"fixture_only":True}
        self.lock={"identity":"fixture_lock","source_identity_sha256":m.canonical(base),
                   "runtime_identity_sha256":m.canonical(base),"asset_identity_sha256":m.canonical(base),
                   "arguments":{"max_det":300},"profiles":[self.profile]}
        self.result={"boxes":np.array([[2,2,8,8]],np.float32),"scores":np.array([.8],np.float32),
                     "classes":np.array([0]),"masks":np.zeros((1,16,16),bool),"native_shape":[16,16],
                     "profile_sha256":self.profile["sha256"],"reference_id":"fixture_ref",
                     "inference_response_available":1,"target_evidence_available":1,"negative_evidence_available":0,
                     "status":"raw_target_candidates","source_identity":base,"runtime_identity":base,
                     "asset_identity":base,"arguments":{"max_det":300},"class_names":{0:"fixture"}}
        self.result["masks"][0,2:8,2:8]=True
        self.cache_patch=mock.patch.object(m,"cache_folder",side_effect=lambda identity:self.folder/"native"/m.canonical(identity))
        self.cache_patch.start()

    def tearDown(self):
        self.cache_patch.stop();self.temp.cleanup()

    def test_exact_cache_resume_and_input_profile_identity(self):
        predictor=mock.Mock()
        predictor.predict.return_value=self.result
        record,reused=m.run_native_task(self.row,self.profile,self.lock,predictor)
        self.assertFalse(reused);self.assertEqual(record["raw_count"],1)
        _,reused=m.run_native_task(self.row,self.profile,self.lock,predictor)
        self.assertTrue(reused);self.assertEqual(predictor.predict.call_count,1)
        self.profile_path.write_bytes(b"changed-profile")
        with self.assertRaisesRegex(ValueError,"identity"):
            m.run_native_task(self.row,self.profile,self.lock,predictor)
        self.assertEqual(predictor.predict.call_count,1)

    def test_failure_is_retained_and_requires_explicit_retry(self):
        predictor=mock.Mock();predictor.predict.side_effect=RuntimeError("synthetic native failure")
        with self.assertRaises(RuntimeError):
            m.run_native_task(self.row,self.profile,self.lock,predictor)
        identity=m.task_identity(self.row,self.profile,self.lock)
        failure=m.cache_folder(identity)/"FAILURE.json"
        self.assertTrue(failure.exists())
        with self.assertRaisesRegex(ValueError,"retry-failed"):
            m.run_native_task(self.row,self.profile,self.lock,predictor)
        self.assertEqual(predictor.predict.call_count,1)
        predictor.predict.side_effect=None;predictor.predict.return_value=self.result
        record,reused=m.run_native_task(self.row,self.profile,self.lock,predictor,retry_failed=True)
        self.assertTrue(record["prior_failure_present"]);self.assertTrue(failure.exists())
        self.assertEqual(len(list(failure.parent.glob("attempt_*/FAILURE.json"))),1)

    def test_raw_mask_precision_and_empty_response_preserved(self):
        for kind in ("binary","float","empty"):
            result=copy.deepcopy(self.result)
            if kind=="float":result["masks"]=result["masks"].astype(np.float32)*.7
            if kind=="empty":
                result.update(boxes=np.empty((0,4),np.float32),scores=np.empty((0,),np.float32),
                              classes=np.empty((0,),np.int64),masks=None)
            dest=self.folder/(kind+".npz");m.save_raw(result,dest);loaded=m.load_raw(dest)
            self.assertTrue(np.array_equal(loaded["boxes"],result["boxes"]))
            if result["masks"] is None:self.assertIsNone(loaded["masks"])
            else:self.assertTrue(np.array_equal(loaded["masks"],result["masks"]))

    def test_cohort_group_geometry_preserves_severe_and_empty_gt(self):
        def inst(index,mask,occ,amodal):
            p=self.folder/"eval"/f"mask{index}.png";Image.fromarray(mask.astype(np.uint8)*255).save(p)
            return {"instance_id":index,"object_id":1,"ignore":False,"mask_visible":str(p.relative_to(ROOT)),
                    "mask_visible_sha256":m.sha(p),"area_visible_px":int(mask.sum()),
                    "area_inframe_amodal_px":amodal,"occlusion_inframe":occ,"bbox_full_xyxy":[1,1,14,14]}
        a=np.zeros((16,16),bool);a[4:6,4:6]=True
        b=np.zeros((16,16),bool)
        row={"sample_id":"FIXTURE_GROUP","split":"eval","image":str(self.image.relative_to(ROOT)),
             "rgb_sha256":m.sha(self.image),"width":16,"height":16,"scene_family":"fixture",
             "group_id":"fixture_group","background_source_id":"fixture",
             "instances":[inst(1,a,.8,20),inst(2,b,1.,20)]}
        result=m.prepare_one(0,row,self.cfg)
        self.assertEqual(result["original_gt_count"],2)
        self.assertIn("severe_occlusion",result["groups"])
        self.assertIsNone(result["pairs"][0]["near_pair"])
        self.assertIsNone(result["pairs"][0]["mask_min_pixel_center_distance_px"])
        self.assertFalse(result["targets"][1]["support_measurable"])

    def test_near_pair_and_merge_duplicate_diagnostics_without_count_cap(self):
        mask=np.zeros((16,16),bool);mask[3:7,3:7]=True
        other=np.zeros_like(mask);other[3:7,8:12]=True
        a={"id":"a","bbox_full_xyxy":[2,2,8,8]}
        b={"id":"b","bbox_full_xyxy":[7,2,13,8]}
        geometry=m.pair_geometry(a,b,mask,other,self.cfg["groups"])
        self.assertTrue(geometry["near_pair"])
        self.assertEqual(geometry["mask_min_pixel_center_distance_px"],2)
        targets=[dict(a,mask=mask),dict(b,mask=other)]
        preds=[{"id":"merged","mask":mask|other},{"id":"copy_a","mask":mask},{"id":"third","mask":mask}]
        diagnostics=m.association_diagnostics(preds,targets,{"scoring_eligible":True,"joint_matches":[]})
        self.assertEqual(diagnostics["two_to_one_merge_candidates"][0]["prediction_id"],"merged")
        self.assertEqual(len(diagnostics["unmatched_published_ids"]),3)
        self.assertIn("NOT automatically non-DJI",diagnostics["unmatched_semantics"])

    def test_fatal_identity_error_stops_worker_after_first_task(self):
        rows=[self.row,dict(self.row,index=8,sample_id="SECOND_FIXTURE")]
        receipt={"cohort_id":"fixture"}
        with mock.patch.object(m,"load_cohort",return_value=(receipt,rows,self.lock)), \
             mock.patch.object(m,"inference_lock",return_value=self.lock), \
             mock.patch.object(m,"runtime_boot",return_value=mock.Mock()), \
             mock.patch.object(m,"run_native_task",side_effect=ValueError("identity fault")) as task, \
             mock.patch.object(m,"write_json") as write:
            with self.assertRaisesRegex(ValueError,"identity fault"):m.run_shard(self.cfg,0)
        self.assertEqual(task.call_count,1)
        self.assertEqual(write.call_args.args[1]["status"],"FATAL_WORKER_STOPPED")

    def test_shards_partition_complete_fixed_sequence(self):
        rows=[{"index":i,"sample_id":str(i)} for i in range(650)]
        shards=[m.shard_rows(rows,i,8) for i in range(8)]
        ids=[r["sample_id"] for shard in shards for r in shard]
        self.assertEqual(len(ids),650);self.assertEqual(len(set(ids)),650)
        self.assertEqual([len(x) for x in shards],[82,82,81,81,81,81,81,81])

    def test_full650_only_and_no_fabricated_gate(self):
        cfg=copy.deepcopy(self.cfg);cfg["dataset"]["manifest"]="data/phrase2/test/annotations.jsonl"
        path=self.folder/"bad_config.json";path.write_text(json.dumps(cfg))
        with self.assertRaisesRegex(ValueError,"test/locked"):m.load_config(path)
        exposed=[{"sample_id":"a"},{"sample_id":"b"}]
        exp=self.folder/"exposed.json";exp.write_text(json.dumps(exposed))
        cfg=copy.deepcopy(self.cfg);cfg["prompt_split"]["exposed_dev_records"]=str(exp.relative_to(ROOT))
        cfg["prompt_split"]["exposed_dev_records_sha256"]=m.sha(exp)
        rows=[{"sample_id":"a","scene_family":"f1"},{"sample_id":"b","scene_family":"f2"},{"sample_id":"c","scene_family":"f1"}]
        split=m.prompt_split(rows,cfg)
        self.assertEqual(split["dev_images"],3);self.assertEqual(split["gate_images"],0)
        self.assertEqual(split["status"],"BLOCKED_NO_UNEXPOSED_SCENE_FAMILY")

    def test_frozen_approved_scorer_identity(self):
        scorer=m.load_scorer(self.cfg)
        self.assertEqual(scorer.JOINT_GATE_CONTRACT,m.JOINT_CONTRACT)
        self.assertEqual(scorer.SCORE_SCHEMA,"action4.yolo_prompt_metrics.v4")
        self.assertIn("predict_snapshot_v002",scorer.__file__)


if __name__=="__main__":
    unittest.main()
