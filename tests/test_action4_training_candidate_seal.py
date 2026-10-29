"""Metadata fixtures only; no model or claimed completed real training."""
import copy,json,unittest
import test_action4_role_seal_metadata as fixtures
from eval.action4_pose_protocol import rank as pose_rank
seal=fixtures.seal

class TrainingSealTests(unittest.TestCase):
    def setUp(self):
        self.base=fixtures.RoleSealMetadataTests();self.base.setUp();self.addCleanup(self.base.tearDown)
        b=self.base;self.root=b.root
        b.cfg.update(role="A0",score_schema="action4.pose_projection_all8_002.v1")
        b.write("snapshot/resolved.json",b.cfg)
        b.spec["files"]["resolved.json"]=seal.digest(b.code/"resolved.json");b.write("spec.json",b.spec)
        cid=seal.canonical(b.cfg)
        for value in (b.train,b.evaluation,b.controller):
            value["config_id"]=cid
            if "role" in value:value["role"]="A0"
        for step in b.evaluation["all_steps"]:
            name=f"output/eval/step_{step:08d}"
            req=b.read(name+".json");req.update(role="A0",config_id=cid);b.write(name+".json",req)
            done=b.read(name+".done.json");done.update(role="A0",config_id=cid,deployment_eligible=step<=40)
            tp={0:0,20:1,40:2,60:4,80:3}[step]
            done["report"].update(score_schema=b.cfg["score_schema"],bbox_only={"per_image":done["report"]["per_image"]},
                pose_gt_count=4,pose_output_count=4,pose_all8_at_002_success=tp,pose_all8_at_002_rate=tp/4,
                pose_fn=4-tp,pose_fp=4-tp,pose_corner_p95_valid_matched_only=None if step==0 else 3.,
                pose_all8_at_002_precision=tp/4,pose_instance_precision=1.,pose_groups={"far":{"gt":1,"tp002":0,"recall":0.}})
            b.write(name+".done.json",done)
            b.write(done["raw_report"],{k:v for k,v in done["report"].items() if k not in ("evaluation_images","evaluation_wall_seconds")})
            if step==40:b.best=done
        b.terminals();self.reports=[]
        self.spec,self.cfg,self.receipts,self.status=seal.inspect("spec.json",report_sink=self.reports)
        self.selection=self.root/"outputs/candidate";(self.selection/"source").mkdir(parents=True)
        worker=self.selection/"source/observer.py";worker.write_text("# metadata fixture, no production execution");worker.chmod(0o444)
        b.write("outputs/candidate/LAUNCH.json",{"run_spec":"spec.json","run_spec_sha256":seal.digest(self.root/"spec.json"),
              "source":str(worker.relative_to(self.root)),"source_sha256":seal.digest(worker)})
        b.write("outputs/candidate/OBSERVER_IDENTITY.json",{"policy":seal.TRAINING_POLICY,"config_id":cid,"run":"output"})
        self.choose(60)
    def choose(self,step):
        b=self.base;done=next(x for x in self.reports if x["step"]==step)
        p=self.selection/"selected"/f"step_{step:08d}.pt";p.parent.mkdir(exist_ok=True);p.write_bytes(b"METADATA FIXTURE NOT MODEL")
        keys=("score_schema","pose_gt_count","pose_output_count","pose_all8_at_002_success","pose_all8_at_002_rate","pose_fn","pose_fp",
              "pose_corner_p95_valid_matched_only","pose_all8_at_002_precision","pose_instance_precision","pose_groups")
        dp=b.out/"eval"/f"step_{step:08d}.done.json"
        self.pointer={"policy":seal.TRAINING_POLICY,"config_id":seal.canonical(self.cfg),"step":step,"sealed":True,
            "purpose":"training_candidate_not_deployment_release","publication_default_changed":False,
            "metrics":{k:done["report"][k] for k in keys},"deployment_guard_passed":done["deployment_eligible"],
            "checkpoint":str(p.relative_to(self.root)),"checkpoint_sha256":seal.digest(p),
            "source_done":{"path":str(dp.relative_to(self.root)),"sha256":seal.digest(dp)}}
        self.publish()
    def publish(self):
        self.base.write("outputs/candidate/TRAINING_BEST.json",self.pointer)
        self.base.write("outputs/candidate/COMPLETE.json",{"status":"CANDIDATE_SELECTION_COMPLETE_PENDING_FINAL_SEAL_AND_RETENTION_AUDIT",
            "training_best":self.pointer,"evaluated_reports":5,"deployment_release":False})
    def inspect(self):
        return seal.inspect_training_selection(self.spec,self.cfg,self.receipts,self.status,self.selection,self.reports,pose_rank)
    def test_all8_winner_sealed_as_training_only(self):
        result=self.inspect()
        self.assertEqual(result["best_step"],60);self.assertEqual(result["legacy_deployment_best_step"],40)
        self.assertFalse(result["deployment_guard_passed"]);self.assertFalse(result["deployment_release"])
        self.assertEqual(self.receipts["BEST.json"]["step"],40)
    def test_lower_available_candidate_cannot_claim_global_best(self):
        self.choose(40)
        with self.assertRaisesRegex(ValueError,"global rank winner"):self.inspect()
    def test_active_or_incomplete_selection_refused(self):
        self.pointer["sealed"]=False;self.publish()
        with self.assertRaises(ValueError):self.inspect()
        self.pointer["sealed"]=True;self.publish()
        value=self.base.read("outputs/candidate/COMPLETE.json");value["evaluated_reports"]=4
        self.base.write("outputs/candidate/COMPLETE.json",value)
        with self.assertRaisesRegex(ValueError,"completely selected"):self.inspect()
    def test_checkpoint_and_metrics_cannot_swap(self):
        self.pointer["metrics"]["pose_all8_at_002_rate"]=.5;self.publish()
        with self.assertRaisesRegex(ValueError,"metrics differ"):self.inspect()
        self.choose(60);(self.root/self.pointer["checkpoint"]).write_bytes(b"wrong weights")
        with self.assertRaisesRegex(ValueError,"checkpoint differs"):self.inspect()
    def test_source_and_separate_deployment_claim_bound(self):
        self.pointer["deployment_guard_passed"]=True;self.publish()
        with self.assertRaisesRegex(ValueError,"Deployment gate"):self.inspect()
        self.choose(60);self.pointer["source_done"]["sha256"]="a"*64;self.publish()
        with self.assertRaisesRegex(ValueError,"report source"):self.inspect()
    def test_import_inventory_rejects_extra_package_and_bytecode(self):
        root=self.base.code/"src/model";root.mkdir(parents=True)
        for name in ("__init__.py","extra.pyc","extra.abi.so"):
            p=root/name;p.write_bytes(b"not executed")
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,"Unlisted"):seal.verify_import_inventory(self.spec)
            p.unlink()
    def test_cli_worker_propagates_candidate_selection(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        with patch.object(seal.subprocess,"run",return_value=SimpleNamespace(stdout="{}")) as run:
            seal.run_worker("spec.json",training_selection="outputs/candidate")
        command=run.call_args.args[0]
        self.assertEqual(command[command.index("--training-candidate-selection")+1],"outputs/candidate")
    def test_real_model_step_check_uses_new_selected_step(self):
        state={"role":"A0","config_id":seal.canonical(self.cfg),"config":self.cfg,"selected_step":60,
               "parent_selected_step":self.cfg["parent_best_step"],"parent_bundle":self.cfg["parent_bundle"]}
        seal.verify_state(state,self.cfg,self.inspect()["best_step"])
        with self.assertRaisesRegex(ValueError,"step mismatch"):seal.verify_state(state,self.cfg,40)
if __name__=="__main__":unittest.main(verbosity=2)
