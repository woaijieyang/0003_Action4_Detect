"""CPU-only v002 configuration/cohort/unknown-GT contracts. No model queries."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np

from eval import action4_full_zeroshot_eval as m

ROOT=Path("/data/zy/vln/a-test")
CFG="configs/eval/action4_full_zeroshot_realref_dev_v002.json"


class RealReferenceRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=m.load_config(CFG)
        cls.root=m.output_root()
        cls.original_v1={name:m.sha(ROOT/"outputs/action4_full_zeroshot_eval_v001/launch_snapshot_v001"/name)
                         for name in ("runner.py","config.json","controller.py")}
        m.activate_snapshot(cls.cfg)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="CPU_FIXTURE_",dir=self.root/"tmp")
        self.folder=Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    @classmethod
    def tearDownClass(cls):
        actual={name:m.sha(ROOT/"outputs/action4_full_zeroshot_eval_v001/launch_snapshot_v001"/name)
                for name in cls.original_v1}
        assert actual==cls.original_v1

    def test_four_shards_are_disjoint_and_exact_1412_budget(self):
        self.assertEqual(self.cfg["inference"]["shard_count"],4)
        rows=[{"index":i,"sample_id":str(i)} for i in range(706)]
        shards=[m.shard_rows(rows,i,4) for i in range(4)]
        self.assertEqual([len(x)*2 for x in shards],[354,354,352,352])
        self.assertEqual(sorted(r["index"] for part in shards for r in part),list(range(706)))
        self.assertEqual(self.cfg["inference"]["startup_sequence"]["initial_shards"],[0,1])
        self.assertEqual(self.cfg["inference"]["startup_sequence"]["additional_shards"],[2,3])

    def test_runtime_v002_only_changes_admitted_resource_metadata(self):
        old=m.read_json("configs/model/action4_yoloe_runtime_gpu1_v001.json")
        new=m.read_json(self.cfg["inference"]["runtime_lock"])
        changed={key for key in set(old)|set(new) if old.get(key)!=new.get(key)}
        self.assertEqual(changed,{"max_concurrent_yolo_workers","root_approval","authority"})
        self.assertEqual(new["max_concurrent_yolo_workers"],4)
        self.assertEqual(new["max_gpu_memory_gib"],8)
        self.assertEqual(new["max_cpu_threads"],4)
        self.assertEqual(m.sha(ROOT/new["authority"]["path"]),new["authority"]["sha256"])
        admitted=m.read_json(new["authority"]["path"])
        self.assertTrue(admitted["GPU0_forbidden"])
        self.assertEqual(admitted["zeroshot_eval_gpu1"]["yolo_max_workers"],4)

    def test_stale_authority_blocks_before_any_GPU_query(self):
        original=m.read_json
        def read(path):
            if str(path)=="configs/execution/action4_active.json":
                return {"active_ref":{"path":"unapproved","sha256":"0"*64}}
            return original(path)
        with mock.patch.object(m,"read_json",side_effect=read),mock.patch.object(m.subprocess,"check_output",side_effect=AssertionError("No GPU query")):
            with self.assertRaisesRegex(ValueError,"authority no longer active"):
                m.runtime_boot(self.cfg)

    def test_actual_declared_two_profiles_are_identity_bound(self):
        code=ROOT/self.cfg["inference"]["prediction_snapshot"]
        snap=m.read_json(code)
        assets=m.read_json(ROOT/snap["code_root"]/self.cfg["inference"]["asset_config"])
        profiles=m.declared_profiles(self.cfg,assets)
        self.assertEqual([x["name"] for x in profiles],["real822_bbox25","real822_mask25"])
        self.assertTrue(all(p["target_class_ids"]==[0] and p["text"] is False for p in profiles))
        self.assertNotEqual(profiles[0]["sha256"],profiles[1]["sha256"])
        bad=copy.deepcopy(self.cfg)
        bad["profile_manifest"]["sha256"]="0"*64
        with self.assertRaisesRegex(ValueError,"identity"):
            m.declared_profiles(bad,assets)

    def test_650_rows_groups_reused_without_any_mask_decode(self):
        with mock.patch("PIL.Image.open",side_effect=AssertionError("GT redecode forbidden")):
            rows,groups,receipt=m.reused_synthetic_cohort(self.cfg)
        self.assertEqual(len(rows),650)
        self.assertEqual(sum(x["original_gt_count"] for x in rows),1300)
        self.assertEqual(groups["image_groups"]["near_pair"],10)
        self.assertEqual(receipt["cohort_sha256"],self.cfg["reuse_cohort"]["cohort_sha256"])

    def test_changed_group_contract_cannot_reuse_old_rows(self):
        bad=copy.deepcopy(self.cfg)
        bad["groups"]["heavy_occlusion_min"]=.3
        with self.assertRaisesRegex(ValueError,"contract changed"):
            m.reused_synthetic_cohort(bad)

    def test_real_query_budget_and_registered_ranges(self):
        ids=m.real_query_ids(self.cfg)
        self.assertEqual(len(ids),56)
        self.assertFalse(set(ids)&set(range(820,825)))
        self.assertEqual((650+len(ids))*2,1412)
        bad=copy.deepcopy(self.cfg);bad["real_dev"]["frame_ranges_inclusive"]=[[900,955]]
        with self.assertRaisesRegex(ValueError,"test/locked"):
            m.real_query_ids(bad)

    def test_enrollment_is_rejected_before_any_query(self):
        bad=copy.deepcopy(self.cfg)
        bad["real_dev"]["frame_ranges_inclusive"]=[[820,875]]
        registry=self.folder/"fake_registered_dev.json"
        registry.write_text(json.dumps({"segments":[{"rows":[{"frame_id":x} for x in range(820,876)]}]}))
        bad["real_dev"]["registered_inventory"]={"path":str(registry),"sha256":m.sha(registry)}
        with self.assertRaisesRegex(ValueError,"Enrollment"):
            m.real_query_ids(bad)

    def test_real_unknown_diagnostics_never_call_scorer(self):
        profile={"name":"real822_bbox25","reference_id":"fixture","sha256":"a"*64,
                 "text":False,"target_class_ids":[0]}
        policy={"id":"fixture","profiles":[profile["name"]],"score_threshold":.25,
                "mask_iou_dedup":.85,"center_distance_small_diag":.05,"prompt_padding_fraction":0}
        row={"sample_id":"real_fixture","domain":"real_harddev_unannotated","frame_id":870,
             "original_gt_count":None,"targets":None,"image":{"sha256":"b"*64},"native_shape":[8,8]}
        record={"raw_sha256":"c"*64,"raw_path":"unused","raw_count":0,"budget_hit":False,"native_status":"unknown_target"}
        raw={"boxes":np.empty((0,4),np.float32),"scores":np.empty(0,np.float32),
             "classes":np.empty(0,np.int64),"masks":None,"native_shape":[8,8]}
        scorer=mock.Mock();scorer.score_image.side_effect=AssertionError("Unknown GT must not be scored")
        with mock.patch.object(m,"completed_native",return_value=record),mock.patch.object(m,"load_raw",return_value=raw):
            result=m.score_one(row,policy,{"profiles":[profile],"identity":"fixture"},self.cfg,scorer)
        self.assertEqual(result["schema"],"action4.full_zeroshot.real_diagnostic.v1")
        self.assertIsNone(result["score"])
        self.assertIsNone(result["original_gt_count"])
        self.assertEqual(result["published_count"],0)
        self.assertEqual(result["accuracy_status"],"NOT_MEASURED_UNKNOWN_GT")
        scorer.score_image.assert_not_called()

    def test_fake_empty_GT_is_rejected_in_real_branch(self):
        profile={"name":"p","reference_id":"fixture","sha256":"a"*64,"text":False}
        policy={"id":"fixture","profiles":["p"],"score_threshold":.25,
                "mask_iou_dedup":.85,"center_distance_small_diag":.05,"prompt_padding_fraction":0}
        row={"sample_id":"real_fixture","domain":"real_harddev_unannotated","frame_id":870,
             "original_gt_count":0,"targets":[],"image":{"sha256":"b"*64},"native_shape":[8,8]}
        record={"raw_sha256":"c"*64,"raw_path":"unused","raw_count":0,"budget_hit":False,"native_status":"unknown_target"}
        raw={"boxes":np.empty((0,4)),"scores":np.empty(0),"classes":np.empty(0),"masks":None}
        with mock.patch.object(m,"completed_native",return_value=record),mock.patch.object(m,"load_raw",return_value=raw):
            with self.assertRaisesRegex(ValueError,"empty GT"):
                m.score_one(row,policy,{"profiles":[profile],"identity":"fixture"},self.cfg,mock.Mock())

    def test_v1_default_delegates_and_v2_cannot_write_v1(self):
        legacy=copy.deepcopy(self.cfg);legacy.pop("profile_manifest")
        command=m.legacy_command(legacy,["status"])
        self.assertEqual(command[2],str(ROOT/"outputs/action4_full_zeroshot_eval_v001/launch_snapshot_v001/runner.py"))
        self.assertIsNone(m.legacy_command(self.cfg,["status"]))
        bad=copy.deepcopy(self.cfg);bad["output_root"]="outputs/action4_full_zeroshot_eval_v001"
        p=self.folder/"bad.json";p.write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError,"baseline output"):
            m.load_config(p)

    def test_previous_incomplete_native_blocks_new_GPU(self):
        folder=self.folder/"prior";folder.mkdir()
        cohort=folder/"rows.jsonl";cohort.write_text(json.dumps({"sample_id":"fixture","index":0})+"\n")
        lock=folder/"lock.json";lock.write_text(json.dumps({"profiles":[{"name":"p"}]}))
        receipt=folder/"receipt.json";receipt.write_text(json.dumps({"contract":{"preparer_source_sha256":"x"},
             "cohort_path":str(cohort),"inference_lock_path":str(lock),"cohort_id":"fixture"}))
        launch=folder/"LAUNCH.json";launch.write_text(json.dumps({"runner_sha256":"x",
             "workers":[{"shard":0,"pid":123,"command":["fake/runner.py"]}]}))
        (folder/"SHARD_00.json").write_text(json.dumps({"status":"IN_PROGRESS","completed_tasks":[],"failures":[]}))
        cfg=copy.deepcopy(self.cfg);cfg["previous_native_release"]={"cohort_receipt":str(receipt),"launch":str(launch),"expected_native_tasks":4550}
        with mock.patch.object(m.subprocess,"run",side_effect=AssertionError("No resource/GPU query after incomplete receipt")):
            with self.assertRaisesRegex(ValueError,"unresolved"):
                m.require_prior_native_release(cfg)


if __name__=="__main__":
    unittest.main(verbosity=2)
