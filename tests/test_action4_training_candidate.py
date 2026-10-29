"""CPU selection fixtures; fake weight bytes are never model/accuracy evidence."""
import copy,json,tempfile,unittest
from pathlib import Path
from model.training.action4_training_candidate import CandidateObserver,canonical,rank,SCHEMA
from eval.action4_pose_protocol import rank as original_rank
R=Path("/data/zy/vln/a-test")
class CandidateTests(unittest.TestCase):
    def setUp(self):
        base=R/"outputs/action4_training_candidate_selection_tests";base.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=base);self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.run=self.root/"run";(self.run/"eval").mkdir(parents=True)
        self.cfg={"role":"A0","mode":"integration","expected_eval_images":2,"output":"run","epochs":1,"expected_units":10,"batch":1,"eval_every":10}
        self.observer=CandidateObserver(self.root,"run","observer",self.cfg)
        (self.run/"BEST.json").write_text('{"unchanged":"deployment_pointer"}')
    def report(self,tp,p95=3.):
        return {"score_schema":SCHEMA,"pose_gt_count":4,"pose_output_count":4,"pose_all8_at_002_success":tp,
            "pose_all8_at_002_rate":tp/4,"pose_fn":4-tp,"pose_fp":4-tp,"pose_corner_p95_valid_matched_only":p95,
            "pose_all8_at_002_precision":tp/4,"pose_instance_precision":1.,
            "pose_groups":{"all":{"gt":4,"tp002":tp,"recall":tp/4},"far":{"gt":1,"tp002":0,"recall":0.}},
            "evaluation_images":2}
    def make(self,step,tp,eligible=False,done=True,weight=True):
        p=self.run/f"eval_step_{step:08d}.pt"
        if weight:p.write_bytes(("fixture-not-real-weights"+str(step)).encode())
        q=self.run/"eval"/f"step_{step:08d}.json"
        q.write_text(json.dumps({"step":step,"role":"A0","config_id":canonical(self.cfg),"snapshot":str(p.relative_to(self.root))}))
        result={"step":step,"role":"A0","config_id":canonical(self.cfg),"deployment_eligible":eligible,"report":self.report(tp)}
        if done:q.with_suffix(".done.json").write_text(json.dumps(result))
        return p,q,result
    def test_high_all8_candidate_survives_failed_deployment_guard(self):
        self.make(0,1,True);self.make(10,3,False)
        self.observer.sweep()
        best=json.loads((self.observer.out/"TRAINING_BEST.json").read_text())
        self.assertEqual(best["step"],10);self.assertFalse(best["deployment_guard_passed"])
        self.assertEqual((self.run/"BEST.json").read_text(),'{"unchanged":"deployment_pointer"}')
        self.assertTrue((self.root/best["checkpoint"]).is_file())
    def test_retains_before_original_worker_unlinks(self):
        p,q,result=self.make(10,3,done=False);self.observer.sweep()
        held=self.observer.out/"pending/step_00000010.pt"
        self.assertTrue(p.samefile(held))
        q.with_suffix(".done.json").write_text(json.dumps(result));p.unlink()
        self.observer.sweep();self.assertEqual(self.observer.best["step"],10)
        self.assertFalse(held.exists())
    def test_missing_historical_weight_is_reported_not_invented(self):
        self.make(0,1,True);self.make(10,4,weight=False)
        self.observer.sweep()
        status=json.loads((self.observer.out/"STATUS.json").read_text())
        self.assertEqual(status["training_best_step"],0)
        self.assertEqual(status["historical_metric_best_step"],10)
        self.assertFalse(status["retained_best_matches_historical_rank"])
        self.assertEqual(status["historical_weights_unavailable"],[10])
    def test_rank_matches_original_and_earlier_tie(self):
        for tp in range(5):
            for p95 in (None,0.,3.):
                report=self.report(tp,p95)
                self.assertEqual(rank(report,10),original_rank(report,10))
        self.make(10,3);self.make(20,3);self.observer.sweep()
        self.assertEqual(self.observer.best["step"],10)
    def test_foreign_or_partial_report_refused(self):
        _,q,result=self.make(0,1)
        result["report"]["evaluation_images"]=1;q.with_suffix(".done.json").write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError,"full development"):self.observer.sweep()
        self.assertFalse((self.observer.out/"TRAINING_BEST.json").exists())
        result["report"]["evaluation_images"]=2;result["config_id"]="foreign"
        q.with_suffix(".done.json").write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError,"Foreign completed"):self.observer.sweep()
    def test_resume_and_terminal_are_training_only(self):
        p,_,_=self.make(0,3);self.make(10,2);self.observer.sweep();p.unlink()
        again=CandidateObserver(self.root,"run","observer",self.cfg)
        again.sweep();self.assertEqual(again.best["step"],0)
        (self.run/"CONTROLLER_COMPLETE.json").write_text(json.dumps({"config_id":canonical(self.cfg),"role":"A0","mode":"integration","attempt_id":"fixture"}))
        (self.run/"EVAL_COMPLETE.json").write_text(json.dumps({"config_id":canonical(self.cfg),"attempt_id":"fixture","evaluations":2,"all_steps":[0,10]}))
        (self.run/"TRAIN_COMPLETE.json").write_text(json.dumps({"step":10,"no_supervision_update_count":0,"attempt_id":"fixture"}))
        self.assertTrue(again.complete_if_terminal())
        result=json.loads((again.out/"COMPLETE.json").read_text())
        self.assertFalse(result["deployment_release"]);self.assertTrue(result["training_best"]["sealed"])
    def test_resume_corrupt_retained_checkpoint_refused(self):
        self.make(0,3);self.observer.sweep()
        (self.root/self.observer.best["checkpoint"]).write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError,"bytes changed"):
            CandidateObserver(self.root,"run","observer",self.cfg)
    def test_stale_terminal_does_not_seal_candidate(self):
        self.make(0,1);self.observer.sweep()
        (self.run/"CONTROLLER_COMPLETE.json").write_text(json.dumps({"config_id":"wrong","role":"A0","mode":"integration"}))
        with self.assertRaisesRegex(ValueError,"Terminal controller"):self.observer.complete_if_terminal()
        self.assertFalse((self.observer.out/"COMPLETE.json").exists())
    def test_invalid_counts_and_null_error_no_advantage(self):
        r=self.report(2);r["pose_fn"]=0
        with self.assertRaises(ValueError):rank(r,0)
        self.assertLess(rank(self.report(2,None),0),rank(self.report(2,3.),0))
if __name__=="__main__":unittest.main(verbosity=2)
