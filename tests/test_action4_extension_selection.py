"""Synthetic CPU cross-run receipts; full model verifier is explicitly mocked."""
import copy,hashlib,json,math,tempfile,unittest
from pathlib import Path
from unittest.mock import patch

from data import action4_extension_selection as metadata
from eval.action4_extension_selection import freeze_global_selection
from model.training.action4_roi_adaptation import validate_recipe,extension_segment
from test_action4_role_seal_metadata import seal as sealer

ROOT=Path("/data/zy/vln/a-test")
BASE=ROOT/"outputs/action4_clip_replay_preparation_v001"


class GlobalSelectionTests(unittest.TestCase):
    def setUp(self):
        BASE.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=BASE);self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name)
        self.parent_cfg=json.loads((ROOT/"runtime/action4_runs/p2_a0_roi15_prediction50_v001/source/resolved_config.json").read_text())
        self.a=self.make_run("incumbent50",self.parent_cfg,50,100,.7,20,20,10.)
        cfg=copy.deepcopy(self.parent_cfg)
        cfg.update(batch=64,micro=2,world_size=4,eval_every=10,formal_epochs=10,formal_training_units=10400,
            schedule_total_updates=1630,parent_bundle=self.a["seal"]["bundle"],
            parent_bundle_sha256=self.a["seal"]["sha256"],parent_best_step=100,
            continuation_segment={"schema":"action4.roi15_best_extension.v1","epochs":10,"batch":64,"eval_every":10,
                                  "parent_seal":self.a["ref"]})
        self.b=self.make_run("predecessor10",cfg,10,0,.7,20,20,10.)

    def ref(self,path):
        return {"path":str(path.relative_to(ROOT)),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}

    def write(self,path,value):
        path.write_text(json.dumps(value,allow_nan=False))
        return self.ref(path)

    def make_run(self,name,cfg,epochs,step,rate,fp,fn,p95):
        out=self.folder/name;out.mkdir();cfg=copy.deepcopy(cfg)
        cfg.update(output=str(out.relative_to(ROOT)),epochs=epochs,mode="full_training")
        config=out/"config.json";self.write(config,cfg)
        spec_ref=self.write(out/"RUN_SPEC.json",{"config":str(config.relative_to(ROOT))})
        bundle=out/"A0.pt";bundle.write_bytes(("SYNTHETIC_NOT_MODEL_"+name).encode())
        report={"score_schema":metadata.SCORE_SCHEMA,"pose_all8_at_002_rate":rate,"pose_fp":fp,"pose_fn":fn,
            "pose_corner_p95_valid_matched_only":p95,"bbox_only":{"per_image":[{"sample_id":f"fixture{i}"} for i in range(650)]},
            "identity":{"A0_step":step,"A0_config_id":metadata._digest(cfg)}}
        best={"role":"A0","config_id":metadata._digest(cfg),"step":step,"report":report,
              "a0_selection_contract":"full_all8_training_candidate_v1"}
        best_ref=self.write(out/"BEST.json",best)
        total=epochs*math.ceil(cfg["expected_units"]/cfg["batch"])
        interval=cfg["eval_every"]
        seal={"schema":"action4.sealed_independent_role.v2","role":"A0","strict_load":True,
            "frozen_backbone_verified":True,"epochs":epochs,"selected_step":step,
            "bundle":str(bundle.relative_to(ROOT)),"sha256":self.ref(bundle)["sha256"],
            "run_spec":spec_ref["path"],"run_spec_sha256":spec_ref["sha256"],"config_id":metadata._digest(cfg),
            "completed_optimizer_steps":total,"fixture_only_not_actual_training":True,
            "completion":{"evaluations":len(set([0,total]+list(range(interval,total+1,interval)))),
                "best_step":step,"config_id":metadata._digest(cfg),
                "evidence_sha256":{best_ref["path"]:best_ref["sha256"]},
                "inference_contract":{"B0_sha256":cfg["evaluation_counterpart_sha256"],
                    "prediction_roi_manifest":cfg["prediction_roi_manifest"],
                    "frozen_eval_b0_predictions":cfg["frozen_eval_b0_predictions"]}}}
        if cfg.get("continuation_segment") is not None:seal["continuation_segment"]=cfg["continuation_segment"]
        ref=self.write(out/"SEALED_ROLE.json",seal)
        return {"seal":seal,"config":cfg,"ref":ref,"best":best,"out":out}

    def revise_best(self,**metrics):
        self.b["best"]["report"].update(metrics)
        best_ref=self.write(self.b["out"]/"BEST.json",self.b["best"])
        self.b["seal"]["completion"]["evidence_sha256"]={best_ref["path"]:best_ref["sha256"]}
        self.b["ref"]=self.write(self.b["out"]/"SEALED_ROLE.json",self.b["seal"])

    def select(self,name="selection.json"):
        def verified(path):
            run=next(r for r in (self.a,self.b) if r["ref"]["path"]==path)
            return {"seal":run["seal"],"config":run["config"],"seal_sha256":run["ref"]["sha256"]}
        with patch("eval.action4_roi_pair_selection._verify_seal",side_effect=verified) as verifier:
            result=freeze_global_selection(self.a["ref"],self.b["ref"],self.folder/name)
        self.assertEqual(verifier.call_count,2)
        return result,self.ref(self.folder/name)

    def config20(self,selection,ref):
        winner=self.b if selection["winner"]=="predecessor" else self.a
        cfg=copy.deepcopy(winner["config"])
        cfg.update(epochs=20,formal_epochs=20,formal_training_units=10400,expected_units=10400,
            batch=64,micro=2,world_size=4,eval_every=10,schedule_total_updates=3260,
            parent_bundle=winner["seal"]["bundle"],parent_bundle_sha256=winner["seal"]["sha256"],
            parent_best_step=winner["seal"]["selected_step"],
            continuation_segment={"schema":"action4.roi15_best_extension.v2","epochs":20,"batch":64,"eval_every":10,
                "parent_seal":selection["winner_seal"],"global_selection":ref})
        return cfg

    def test_exact_tie_retains_incumbent_despite_predecessor_step_zero_and_runs_twenty(self):
        result,ref=self.select()
        self.assertEqual(result["winner"],"incumbent")
        cfg=self.config20(result,ref)
        validate_recipe(cfg)
        self.assertTrue(sealer.validate_formal_recipe(cfg))
        self.assertEqual(cfg["epochs"],20)
        self.assertEqual(len(set([0,3260]+list(range(10,3261,10)))),327)
        self.assertFalse(result["local_optimizer_step_used_for_cross_run_rank"])

    def test_all8_then_fpfns_then_p95_priority(self):
        for index,(metrics,expected) in enumerate([
            ({"pose_all8_at_002_rate":.71,"pose_fp":100,"pose_fn":100,"pose_corner_p95_valid_matched_only":100.},"predecessor"),
            ({"pose_all8_at_002_rate":.7,"pose_fp":10,"pose_fn":20,"pose_corner_p95_valid_matched_only":100.},"predecessor"),
            ({"pose_all8_at_002_rate":.7,"pose_fp":20,"pose_fn":20,"pose_corner_p95_valid_matched_only":9.},"predecessor"),
            ({"pose_all8_at_002_rate":.69,"pose_fp":0,"pose_fn":0,"pose_corner_p95_valid_matched_only":1.},"incumbent")]):
            self.revise_best(**metrics)
            result,ref=self.select(f"selection{index}.json")
            self.assertEqual(result["winner"],expected)
            validate_recipe(self.config20(result,ref))
            self.assertTrue(sealer.validate_formal_recipe(self.config20(result,ref)))

    def test_missing_incomplete_predecessor_cannot_prepare_twenty(self):
        self.b["seal"]["completion"]["evaluations"]-=1
        self.b["ref"]=self.write(self.b["out"]/"SEALED_ROLE.json",self.b["seal"])
        with self.assertRaisesRegex(ValueError,"coverage"):self.select()
        self.assertFalse((self.folder/"selection.json").exists())
        missing={"path":str((self.folder/"nonexistent.json").relative_to(ROOT)),"sha256":"a"*64}
        with self.assertRaises(FileNotFoundError):freeze_global_selection(self.a["ref"],missing,self.folder/"never.json")

    def test_bound_report_tamper_wrong_winner_and_unregistered_budget_rejected(self):
        result,ref=self.select()
        changed=copy.deepcopy(result);changed["winner_seal"]=self.b["ref"]
        wrong=self.write(self.folder/"wrong.json",changed)
        with self.assertRaises(ValueError):metadata.validate_global_selection(wrong)
        cfg=self.config20(result,ref)
        for key,value in (("epochs",21),("schedule_total_updates",3270),("parent_best_step",1)):
            bad=copy.deepcopy(cfg);bad[key]=value
            with self.assertRaises(ValueError):validate_recipe(bad)
        self.b["best"]["report"]["pose_all8_at_002_rate"]=.99
        self.write(self.b["out"]/"BEST.json",self.b["best"])
        with self.assertRaises(ValueError):metadata.validate_global_selection(ref)

    def test_changed_frozen_B0_source_and_alien_predecessor_rejected(self):
        cfg=copy.deepcopy(self.b["config"]);cfg["evaluation_counterpart_sha256"]="f"*64
        alien=self.make_run("alien10",cfg,10,0,.8,10,10,1.)
        with self.assertRaisesRegex(ValueError,"data/B0"):
            metadata._resolve(self.a["ref"],alien["ref"])
        cfg=copy.deepcopy(self.b["config"]);cfg["continuation_segment"]["parent_seal"]={"path":"alien","sha256":"c"*64}
        alien=self.make_run("wrongparent10",cfg,10,0,.8,10,10,1.)
        with self.assertRaisesRegex(ValueError,"child"):
            metadata._resolve(self.a["ref"],alien["ref"])

    def test_rank_matches_existing_pose_protocol_without_local_step(self):
        from eval.action4_pose_protocol import rank
        report=self.a["best"]["report"]
        self.assertEqual(metadata.comparison_key(report),rank(report,999)[:3])



    def test_twenty_pair_requires327_evals_and_integration_keeps3260_budget(self):
        from eval.action4_roi_pair_selection import validate_pair
        result,ref=self.select()
        cfg=self.config20(result,ref)
        contract={"roi_expansion":1.5,"B0_sha256":cfg["evaluation_counterpart_sha256"],
            "frozen_eval_b0_predictions":cfg["frozen_eval_b0_predictions"],
            "continuation_segment":cfg["continuation_segment"]}
        a={"role":"A0","strict_load":True,"frozen_backbone_verified":True,"epochs":20,
           "completed_optimizer_steps":3260,"continuation_segment":cfg["continuation_segment"],
           "score_schema":cfg["score_schema"],"completion":{"evaluations":327,"inference_contract":contract}}
        b={"role":"B0","strict_load":True,"sha256":cfg["evaluation_counterpart_sha256"]}
        validate_pair(a,cfg,b,{"role":"B0"})
        wrong=copy.deepcopy(a);wrong["completion"]["evaluations"]=326
        with self.assertRaises(ValueError):validate_pair(wrong,cfg,b,{"role":"B0"})
        integration=copy.deepcopy(cfg)
        integration.update(mode="integration",epochs=1,expected_units=1312,train_indices=list(range(1312)))
        validate_recipe(integration)
        self.assertEqual(integration["schedule_total_updates"],3260)
        with self.assertRaises(ValueError):sealer.validate_formal_recipe(integration)


if __name__=="__main__":
    unittest.main()
