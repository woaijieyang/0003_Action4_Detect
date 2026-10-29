"""Small async state-machine fixtures, never training/evaluation/model accuracy."""
import copy,json,hashlib,types,sys,unittest,uuid
from pathlib import Path
from unittest.mock import patch
from data.action4_paths import PROJECT_ROOT as R
from model.training.action4_runtime import atomic_json,canonical_hash
from model.training.action4_async_recovery import commit_result
from model.training.action4_best_only_retention import prune_old_selected,finalize_best_only
from model.training.action4_stage2_dispatch import finalize_evaluations
from eval.action4_stage2_eval import prepare_corner_selection,rank_result,evaluate

ROOT=R/"outputs/action4_corner_async_preparation_v001"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def binding(p):return {"path":str(p.relative_to(R)),"sha256":sha(p)}
def report(tp=2,difficult=1):
    successes=[i<difficult if i<2 else i-2<tp-difficult for i in range(4)]
    return {"score_schema":"action4.pose_projection_all8_002.v1","pose_gt_count":4,"pose_output_count":4,
        "pose_all8_at_002_success":tp,"pose_all8_at_002_rate":tp/4,"pose_all8_at_005_rate":1.,
        "pose_all8_at_002_precision":tp/4,"pose_all8_at_002_f1":tp/4,"pose_fp":4-tp,"pose_fn":4-tp,
        "invalid_pose_count":0,"pose_instance_precision":1.,"pose_corner_p95_valid_matched_only":3. if tp<4 else 1.,
        "strict_image_success_rate":float(tp==4),
        "per_gt_failure_reason":[{"sample_id":"fixture","instance_id":i,"prediction_id":"p"+str(i),
            "all8_at_002":ok,"all8_at_005":True,"errors_px":[1. if ok else 3.]*8,"failure_reason":None if ok else "corner_error"}
            for i,ok in enumerate(successes)],"unmatched_pose_predictions":[],
        "pose_groups":{"all":{"gt":4,"tp002":tp,"recall":tp/4},"difficult":{"gt":2,"tp002":difficult,"recall":difficult/2}}}


class CornerAsyncTests(unittest.TestCase):
    def setUp(self):
        self.out=ROOT/uuid.uuid4().hex;self.out.mkdir(parents=True);(self.out/"eval").mkdir()
        source=self.out/"SOURCE.json";atomic_json(source,{"scope":"async unit fixture; no model execution"})
        parent=self.out/"parent.pt";parent.write_bytes(b"engineering parent weights placeholder")
        self.cfg={"role":"A0","mode":"integration","a0_variant":"a0_corner_image","corner_enabled":True,
            "sam_enabled":True,"corner_eval_source":binding(source),"parent_bundle":str(parent.relative_to(R)),
            "parent_bundle_sha256":sha(parent),"parent_best_step":3,
            "epochs":1,"expected_units":2,"batch":1,"eval_every":1,"expected_eval_images":1}
        self.ref=None

    def make(self,step,quality,raw=None,candidate=None):
        raw=report() if raw is None else raw;candidate=report(3,1) if candidate is None else candidate
        snapshot=self.out/f"eval_step_{step:08d}.pt";snapshot.write_bytes(("engineering fixture weights "+str(step)).encode())
        dest=self.out/f"result_{step}";dest.mkdir()
        pred=dest/"PREDICTIONS.json";atomic_json(pred,{"scope":"source fixture only","step":step})
        cal=dest/"CALIBRATION.json";atomic_json(cal,{
            "schema":"action4.corner_checkpoint_calibration.v1","quality_ready":quality,
            "quality_gate_passed_for_selection":quality,
            "run_binding":{"checkpoint_sha256":sha(snapshot),"source_sha256":self.cfg["corner_eval_source"]["sha256"],"predictions_sha256":sha(pred)},
            "raw_report":{"pose_report":raw},"candidate_report":{"pose_report":candidate},
            "final_path":"candidate" if quality else "raw"})
        rp=dest/"RAW.json";cp=dest/"CANDIDATE.json";atomic_json(rp,raw);atomic_json(cp,candidate)
        result_report=dict(copy.deepcopy(candidate if quality else raw),
            corner_quality_ready=quality,quality_gate_passed_for_selection=quality,
            corner_eval_source=copy.deepcopy(self.cfg["corner_eval_source"]),calibration_receipt=binding(cal),
            prediction_artifact=binding(pred),raw_pose_report=copy.deepcopy(raw),candidate_pose_report=copy.deepcopy(candidate),
            raw_pose_report_artifact=binding(rp),candidate_pose_report_artifact=binding(cp))
        fields,reference=prepare_corner_selection(self.cfg,step,snapshot,result_report,self.ref)
        if step==0:
            self.ref=reference;atomic_json(self.out/"REFERENCE_REPORT.json",reference)
        result={"step":step,"role":"A0","config_id":canonical_hash(self.cfg),"training_mode":"integration","report":result_report,
            "raw_report":str((dest/"REPORT.json").relative_to(R)),**fields}
        atomic_json(dest/"REPORT.json",result_report)
        result["report_artifact"]=binding(dest/"REPORT.json")
        request=self.out/"eval"/f"step_{step:08d}.json"
        atomic_json(request,{"step":step,"role":"A0","config_id":canonical_hash(self.cfg),"snapshot":str(snapshot.relative_to(R))})
        return request,snapshot,result

    def finish(self):
        terminal={"step":2,"attempt_id":"attempt-fixture","no_supervision_update_count":0}
        atomic_json(self.out/"TRAIN_COMPLETE.json",terminal)
        (self.out/"last.pt").write_bytes(b"owned recovery fixture")
        return finalize_evaluations(self.cfg,self.out,"attempt-fixture",terminal)

    def test_all_unready_no_fake_best_and_parent_fallback(self):
        zero=self.make(0,False)
        high=self.make(1,False,raw=report(4,2))
        late=self.make(2,False,raw=report(4,2))
        # Out-of-order completions: high raw scores cannot bypass quality admission.
        for request,snapshot,result in (late,zero,high):
            commit_result(self.out,request,snapshot,result,rank_result)
            self.assertTrue(snapshot.exists());self.assertTrue(request.with_suffix(".done.json").exists())
            self.assertFalse((self.out/"BEST.json").exists());self.assertEqual(prune_old_selected(str(self.out)),[])
        self.assertFalse((self.out/"best.pt").exists());self.assertEqual(list(self.out.glob("selected_step_*.pt")),[])
        done=self.finish()
        self.assertEqual(done["status"],"NO_DEPLOYABLE_CHECKPOINT");self.assertIsNone(done["best_step"])
        ev=json.loads((self.out/"EVAL_COMPLETE.json").read_text())
        self.assertEqual(ev["evaluations"],3);self.assertEqual(ev["all_steps"],[0,1,2])
        self.assertEqual(ev["deployment_eligible_evaluations"],0)
        self.assertEqual(done["parent_stage_fallback"]["sha256"],self.cfg["parent_bundle_sha256"])
        self.assertTrue((self.out/"last.pt").exists())
        retention=json.loads((self.out/"CORNER_RETENTION.json").read_text())
        self.assertFalse(retention["cleanup_performed"])
        self.assertTrue(retention["provisional_dependency_hold"])
        self.assertGreater(retention["artifact_bytes"]["unique_inode_logical_bytes"],0)
        self.assertEqual(len(retention["retirement_review_candidates"]),3)
        self.assertFalse(done["accuracy_accepted"])
        self.assertEqual(finalize_best_only(str(self.out),remove_recovery=True),[])
        self.assertTrue((self.out/"last.pt").exists())

    def test_ready_but_original_group_guard_fails(self):
        zero=self.make(0,False);request,snapshot,result=self.make(1,True,candidate=report(2,0))
        self.assertTrue(result["report"]["quality_gate_passed_for_selection"])
        self.assertFalse(result["pose_protocol_eligible"]);self.assertFalse(result["deployment_eligible"])
        commit_result(self.out,*zero,rank_result)
        commit_result(self.out,request,snapshot,result,rank_result)
        self.assertFalse((self.out/"BEST.json").exists())
        with self.assertRaises(ValueError):rank_result(result)

    def test_eligible_winner_original_rank_earlier_tie_and_recovery_refs(self):
        zero=self.make(0,False);one=self.make(1,True);two=self.make(2,True)
        for row in (two,zero,one):commit_result(self.out,*row,rank_result)
        best=json.loads((self.out/"BEST.json").read_text());self.assertEqual(best["step"],1)
        self.assertTrue((self.out/"best.pt").samefile(self.out/"selected_step_00000001.pt"))
        self.assertTrue(two[1].exists());self.assertTrue(zero[1].exists())
        self.assertTrue((self.out/"selected_step_00000002.pt").exists())
        # Replay is safe and keeps source paths used by prediction/calibration artifacts.
        commit_result(self.out,*one,rank_result)
        self.assertEqual(json.loads((self.out/"BEST.json").read_text())["step"],1)
        done=self.finish();self.assertEqual(done["status"],"COMPLETE_WITH_DEPLOYABLE_CHECKPOINT")
        self.assertEqual(done["best_step"],1)
        ev=json.loads((self.out/"EVAL_COMPLETE.json").read_text())
        self.assertEqual(ev["deployment_eligible_evaluations"],2);self.assertEqual(ev["rejected_evaluations"],1)
        retained=json.loads((self.out/"CORNER_RETENTION.json").read_text())
        self.assertEqual(len(retained["evaluated_snapshot_bindings"]),3)
        self.assertTrue((self.out/"last.pt").exists())

    def test_step0_reference_uses_raw_not_candidate(self):
        request,snapshot,result=self.make(0,True,raw=report(2,1),candidate=report(4,2))
        self.assertEqual(self.ref["pose_all8_at_002_success"],2)
        self.assertEqual(result["report"]["pose_all8_at_002_success"],4)

    def test_mixed_config_source_and_false_eligibility_rejected(self):
        zero=self.make(0,False);commit_result(self.out,*zero,rank_result)
        one=self.make(1,True)
        bad=copy.deepcopy(one[2]);bad["config_id"]="a"*64
        with self.assertRaises(ValueError):commit_result(self.out,one[0],one[1],bad,rank_result)
        bad=copy.deepcopy(one[2]);bad["deployment_eligible"]=False
        with self.assertRaises(ValueError):commit_result(self.out,one[0],one[1],bad,rank_result)
        source=self.out/"OTHER_SOURCE.json";atomic_json(source,{"different":"code"})
        bad=copy.deepcopy(one[2]);bad["corner_eval_source"]=binding(source);bad["report"]["corner_eval_source"]=binding(source)
        with self.assertRaisesRegex(ValueError,"Mixed|immutable report"):commit_result(self.out,one[0],one[1],bad,rank_result)
        bad_report=copy.deepcopy(one[2]["report"]);bad_report["corner_eval_source"]=binding(source)
        with self.assertRaises(ValueError):prepare_corner_selection(self.cfg,1,one[1],bad_report,self.ref)
        changed=self.out/"bad_snapshot.pt";changed.write_bytes(b"different bytes")
        with self.assertRaises(ValueError):prepare_corner_selection(self.cfg,1,changed,one[2]["report"],self.ref)

    def test_corner_evaluate_calls_new_runner_not_old_staged_path(self):
        _,snapshot,result=self.make(0,False)
        calls=[];module=types.ModuleType("eval.action4_image_guided_eval")
        def run(cfg,path,destination,data,device):
            calls.append((cfg,path,destination,data,device));return result["report"]
        module.run_corner_evaluation=run;data=object();destination=self.out/"evaluation_callback"
        with patch.dict(sys.modules,{"eval.action4_image_guided_eval":module}),patch("eval.action4_stage2_eval.run_staged",side_effect=AssertionError("old path forbidden")):
            actual=evaluate(self.cfg,snapshot,destination,data,device="cpu")
        self.assertEqual(actual,result["report"]);self.assertEqual(calls[0][-1],"cpu")
        self.assertIs(calls[0][3],data);self.assertTrue((destination/"REPORT.json").exists())

    def test_corner_crash_before_done_replays_without_losing_snapshot(self):
        import model.training.action4_async_recovery as recovery
        row=self.make(0,True)
        original=recovery.atomic_json
        def crash(path,value):
            if str(path).endswith(".done.json"):raise RuntimeError("injected before done")
            return original(path,value)
        with patch.object(recovery,"atomic_json",side_effect=crash):
            with self.assertRaisesRegex(RuntimeError,"injected"):commit_result(self.out,*row,rank_result)
        self.assertTrue(row[1].exists());self.assertFalse(row[0].with_suffix(".done.json").exists())
        commit_result(self.out,*row,rank_result)
        self.assertTrue(row[1].exists());self.assertTrue(row[0].with_suffix(".done.json").exists())
        self.assertEqual(json.loads((self.out/"BEST.json").read_text())["step"],0)

    def test_completion_requires_every_report_and_same_source(self):
        zero=self.make(0,False);one=self.make(1,False)
        for row in (zero,one):commit_result(self.out,*row,rank_result)
        terminal={"step":2,"attempt_id":"attempt-fixture","no_supervision_update_count":0}
        atomic_json(self.out/"TRAIN_COMPLETE.json",terminal)
        with self.assertRaisesRegex(ValueError,"Incomplete"):finalize_evaluations(self.cfg,self.out,"attempt-fixture",terminal)
        two=self.make(2,False);commit_result(self.out,*two,rank_result)
        completed=json.loads(two[0].with_suffix(".done.json").read_text());completed["config_id"]="a"*64
        atomic_json(two[0].with_suffix(".done.json"),completed)
        with self.assertRaises(ValueError):finalize_evaluations(self.cfg,self.out,"attempt-fixture",terminal)
        self.assertFalse((self.out/"EVAL_COMPLETE.json").exists())

    def test_post_bind_report_mutation_rejected_before_best_or_done(self):
        row=self.make(0,True)
        row[2]["report"]["pose_all8_at_002_rate"]=.99
        with self.assertRaisesRegex(ValueError,"immutable report"):
            commit_result(self.out,*row,rank_result)
        self.assertFalse((self.out/"BEST.json").exists())
        self.assertFalse(row[0].with_suffix(".done.json").exists())

    def test_report_binding_required_and_terminal_done_tamper_rejected(self):
        row=self.make(0,False)
        missing=copy.deepcopy(row[2]);missing.pop("report_artifact")
        with self.assertRaisesRegex(ValueError,"report path/binding"):
            commit_result(self.out,row[0],row[1],missing,rank_result)
        for step in range(3):
            value=row if step==0 else self.make(step,False)
            commit_result(self.out,*value,rank_result)
        done=row[0].with_suffix(".done.json");value=json.loads(done.read_text())
        value["report"]["pose_all8_at_002_rate"]=.99;atomic_json(done,value)
        with self.assertRaisesRegex(ValueError,"immutable report"):self.finish()
        self.assertFalse((self.out/"EVAL_COMPLETE.json").exists())

    def test_legacy_B0_commit_semantics_unchanged(self):
        snapshot=self.out/"legacy.pt";snapshot.write_bytes(b"legacy fixture")
        request=self.out/"eval/legacy.json";atomic_json(request,{})
        result={"role":"B0","step":0,"report":{"mAP50_95":.5,"overall":{"f1":.6}}}
        commit_result(self.out,request,snapshot,result,rank_result)
        self.assertTrue((self.out/"BEST.json").exists());self.assertTrue((self.out/"best.pt").exists())
        self.assertFalse(snapshot.exists())


if __name__=="__main__":unittest.main(verbosity=2)
