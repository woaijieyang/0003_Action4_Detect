import hashlib,json,tempfile,unittest
from pathlib import Path
from eval.action4_video_contract import resource_plan,validate_selection

class VideoContractTests(unittest.TestCase):
    def authority(self):
        return dict(authorized_evaluation_gpu=[4,5,6,7],forbidden_gpu_ids=[0,1,2,3],cpu_policy=dict(allowed_logical_cpu_ids=list(range(24,96))),cpu_budget_limit=72)
    def test_gpu1_rejected(self):
        with self.assertRaises(ValueError):resource_plan([1],list(range(90,96)),0,self.authority())
    def test_tail_allocation_and_budget(self):
        self.assertEqual(resource_plan([7],list(range(90,96)),64,self.authority()),[list(range(90,96))])
        with self.assertRaises(ValueError):resource_plan([7],list(range(90,96)),65,self.authority())
    def test_affinity_low_or_duplicate_rejected(self):
        for cpus in (list(range(6)),[90]*6):
            with self.assertRaises(ValueError):resource_plan([7],cpus,0,self.authority())
    def test_temporal_claim_rejected(self):
        with self.assertRaisesRegex(ValueError,"continuous"):
            validate_selection(dict(temporal_modules=["SAM","CoTracker"]))
    def test_frozen_selection_and_tampering(self):
        with tempfile.TemporaryDirectory(dir="outputs/action4_video_selection_preparation_v001") as tmp:
            root=Path(tmp).resolve();cfg={"role_seals":{}}
            for role in ("A0","B0"):
                bundle=root/(role+".pt");bundle.write_bytes(role.encode());sha=hashlib.sha256(role.encode()).hexdigest()
                seal=root/(role+".json");seal.write_text(json.dumps(dict(schema="action4.sealed_independent_role.v2",strict_load=True,role=role,sha256=sha,bundle=bundle.name)))
                cfg[role.lower()+"_bundle"]=bundle.name;cfg[role.lower()+"_sha256"]=sha;cfg["role_seals"][role]=seal.name
            self.assertEqual(validate_selection(cfg,root)["a0_sha256"],cfg["a0_sha256"])
            (root/"A0.pt").write_bytes(b"wrong")
            with self.assertRaises(ValueError):validate_selection(cfg,root)
    def test_default_rejected(self):
        with tempfile.TemporaryDirectory(dir="outputs/action4_video_selection_preparation_v001") as tmp:
            root=Path(tmp).resolve();(root/"models/default").mkdir(parents=True);(root/"models/default/A0.pt").write_bytes(b"a")
            with self.assertRaisesRegex(ValueError,"defaults"):
                validate_selection(dict(a0_bundle="models/default/A0.pt"),root)
    def test_cache_reuse_rejects_rgb_and_time_changes(self):
        from eval.action4_video_contract import cached_prediction
        with tempfile.TemporaryDirectory(dir="outputs/action4_video_selection_preparation_v001") as tmp:
            path=Path(tmp)/"rgb";path.write_bytes(b"rgb")
            row=dict(frame_id=0,pts=0,time_base="1/11988",rgb_sha256=hashlib.sha256(b"rgb").hexdigest(),prediction={"instances":[]})
            self.assertEqual(cached_prediction({0:row},dict(frame_id=0,pts=0),"1/11988",path),{"instances":[]})
            with self.assertRaises(ValueError):cached_prediction({0:row},dict(frame_id=0,pts=400),"1/11988",path)
            path.write_bytes(b"changed")
            with self.assertRaises(ValueError):cached_prediction({0:row},dict(frame_id=0,pts=0),"1/11988",path)
    def test_cache_model_mismatch_rejected_before_rows(self):
        from eval.action4_video_contract import load_prediction_cache
        with tempfile.TemporaryDirectory(dir="outputs/action4_video_selection_preparation_v001") as tmp:
            root=Path(tmp).resolve();p=root/"cache.json"
            p.write_text(json.dumps(dict(schema="action4.baseline_video_prediction_cache.v1",temporal_mode="independent_frames_baseline",selected_bundles={"a0_sha256":"old"})))
            with self.assertRaisesRegex(ValueError,"identities differ"):load_prediction_cache(p,{"a0_sha256":"new"},root)
if __name__=="__main__": unittest.main()
