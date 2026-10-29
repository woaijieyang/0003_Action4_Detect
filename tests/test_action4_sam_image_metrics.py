"""CPU scoring fixtures; no native inference or GT-based output changes."""
import os
import unittest
import uuid
from unittest.mock import patch
import numpy as np
from eval.action4_sam_image_metrics import (
    PROJECT_ROOT, score_mask_set, score_image, summarize, upstream_assignment, load_completed_job, cpu_affinity_receipt, validate_native_domain, paired_conditional, full_conditional_domain,
)

class SamImageMetricsTest(unittest.TestCase):
    def setUp(self):
        self.a=np.zeros((8,10),bool);self.a[1:5,1:4]=True
        self.b=np.zeros((8,10),bool);self.b[1:5,6:9]=True
        self.targets=[{"id":"A","annotation_status":"valid","mask":self.a,"bbox_full_xyxy":[1,1,4,5]},
                      {"id":"B","annotation_status":"valid","mask":self.b,"bbox_full_xyxy":[6,1,9,5]}]
        self.candidates=[{"candidate_id":"ca","eligible_for_SAM_diagnostic_prompt":True,"bbox_full_native_xyxy":[1,1,4,5]},
                         {"candidate_id":"cb","eligible_for_SAM_diagnostic_prompt":True,"bbox_full_native_xyxy":[6,1,9,5]}]
    def job(self,cid,raw,bound=None,status="target_present"):
        return {"candidate_id":cid,"status":status,"available":bound is not None,
                "bound_prediction_id":bound,"raw_predictions":raw}
    def meta(self,id,targets=None):
        targets=self.targets if targets is None else targets
        return {"sample_id":id,"groups":["all_dev"],"original_gt_count":len(targets),
                "targets":[dict(t,support_measurable=t.get("mask") is not None and bool(t["mask"].any()),
                                 visible_pixels=int(t["mask"].sum()) if t.get("mask") is not None else 0) for t in targets]}
    def test_raw_good_mask_never_replaces_wrong_frozen_binding(self):
        raw=[{"id":"good","mask":self.a},{"id":"selected-other","mask":self.b}]
        job=self.job("ca",raw,"selected-other")
        r=score_image(self.candidates,[job],self.targets,image_id="image")
        self.assertEqual(r["raw_native_diagnostic"]["coverage_tp"],2)
        self.assertEqual(r["frozen_bound_masks"]["coverage_tp"],1)
        fixed=next(p for p in r["conditional"] if p["candidate_id"]=="ca")
        self.assertEqual(fixed["gt_id"],"A");self.assertEqual(fixed["iou"],0)
        self.assertEqual(fixed["selected_prediction_id"],"selected-other")
        self.assertEqual(fixed["wrong_instance_support_gt_ids"],["B"])
        self.assertEqual(job["bound_prediction_id"],"selected-other")
        self.assertFalse(r["GT_used_to_change_native_binding"])
    def test_unknown_response_and_pending_remain_conditional_domain(self):
        r=score_image(self.candidates,[self.job("ca",[],None,"target_visibility_unknown")],self.targets,image_id="image")
        self.assertEqual(len(r["conditional"]),2)
        a,b=r["conditional"]
        self.assertTrue(a["response_completed"]);self.assertFalse(a["mask_available"])
        self.assertFalse(b["response_completed"]);self.assertFalse(b["mask_available"])
        self.assertFalse(r["image_transport_complete"])
        self.assertEqual(r["frozen_bound_masks"]["coverage_tp"],0)
    def test_merged_mask_cannot_count_as_two_supported_targets(self):
        raw=[{"id":"merged","mask":self.a|self.b}]
        r=score_mask_set(raw,self.targets)
        self.assertEqual(r["coverage_tp"],1)
        self.assertEqual(r["strict_support_tp"],0)
        self.assertEqual(r["diagnostics"][0]["category"],"merged_support")
        self.assertEqual(r["diagnostics"][0]["merged_gt_ids"],["A","B"])
    def test_duplicate_selected_masks_not_double_success(self):
        jobs=[self.job("ca",[{"id":"first","mask":self.a}],"first"),
              self.job("cb",[{"id":"copy","mask":self.a.copy()}],"copy")]
        r=score_image(self.candidates,jobs,self.targets,image_id="image")
        self.assertEqual(r["frozen_bound_masks"]["strict_support_tp"],1)
        self.assertEqual(len(r["frozen_bound_masks"]["duplicate_gt_support"]["A"]),2)
        self.assertEqual(len(r["frozen_bound_masks"]["exact_duplicate_mask_pairs"]),1)
    def test_unknown_annotation_has_no_fake_precision_or_conditional_score(self):
        targets=[self.targets[0],dict(self.targets[1],annotation_status="unknown",mask=None)]
        r=score_image(self.candidates,[self.job("ca",[{"id":"p","mask":self.a}],"p")],targets,image_id="image")
        self.assertFalse(r["frozen_bound_masks"]["annotation_domain_eligible"])
        self.assertIsNone(r["frozen_bound_masks"]["coverage_tp"])
        self.assertIsNone(r["frozen_bound_masks"]["class_precision"])
        summary=summarize([self.meta("image",targets)],[],[r],
            {"image":upstream_assignment(self.candidates,targets)},
            [{"task_key":"a","image":{"image_id":"image"}},{"task_key":"b","image":{"image_id":"image"}}],{"a","b"})["all"]
        self.assertEqual(summary["original_gt_denominator"],2)
        self.assertEqual(summary["conditional_expected_upstream_matched_prompts"],0)
        self.assertIsNone(summary["bound_coverage_recall"])
        self.assertIsNone(summary["availability_weighted_conditional_IoU"])
    def test_no_visible_GT_is_unmeasured_not_target_absent(self):
        targets=[self.targets[0],dict(self.targets[1],mask=np.zeros_like(self.b))]
        r=score_mask_set([{"id":"p","mask":self.a}],targets)
        self.assertEqual(r["original_gt_count"],2)
        self.assertEqual(r["visible_support_gt_count"],1)
        self.assertEqual(r["support_unavailable_gt_ids"],["B"])
    def test_background_is_unlabeled_not_non_DJI_proof(self):
        mask=np.zeros((8,10),bool);mask[7,0]=True
        r=score_mask_set([{"id":"cloth-like","mask":mask}],self.targets)
        self.assertEqual(r["diagnostics"][0]["category"],"background_or_unlabeled")
        self.assertFalse(r["diagnostics"][0]["non_DJI_class_asserted"])
        self.assertIsNone(r["class_precision"])
    def test_background_spill_is_not_reverse_pair_cross_target_leakage(self):
        spill=self.a.copy();spill[6,0:4]=True
        r=score_mask_set([{"id":"spill","mask":spill}],self.targets)
        self.assertEqual(r["diagnostics"][0]["category"],"target_plus_background_contamination")
        # A genuine nontrivial second-target overlap remains cross-target.
        cross=self.a.copy();cross[1,6]=True
        r=score_mask_set([{"id":"cross","mask":cross}],self.targets)
        self.assertEqual(r["diagnostics"][0]["category"],"cross_target_contamination")

    def test_stable_ID_reordering_does_not_change_matches(self):
        ps=[{"id":"pa","mask":self.a},{"id":"pb","mask":self.b}]
        a=score_mask_set(ps,self.targets);b=score_mask_set(ps[::-1],self.targets[::-1])
        self.assertEqual(a["strict_support_matches"],b["strict_support_matches"])
        self.assertEqual(upstream_assignment(self.candidates,self.targets),upstream_assignment(self.candidates[::-1],self.targets[::-1]))
    def test_partial_and_unprompted_do_not_shrink_full_denominator(self):
        jobs=[self.job("ca",[{"id":"pa","mask":self.a}],"pa")]
        r=score_image(self.candidates,jobs,self.targets,image_id="one")
        other=[self.candidates[0],dict(self.candidates[1],eligible_for_SAM_diagnostic_prompt=False)]
        cohort=[self.meta("one"),self.meta("two")]
        assignments={"one":upstream_assignment(self.candidates,self.targets),"two":upstream_assignment(other,self.targets)}
        expected=[{"task_key":"one-a","image":{"image_id":"one"}},{"task_key":"one-b","image":{"image_id":"one"}},{"task_key":"two-a","image":{"image_id":"two"}}]
        s=summarize(cohort,[],[r],assignments,expected,{"one-a"})["all"]
        self.assertEqual(s["original_gt_denominator"],4)
        self.assertEqual(s["expected_prompt_jobs"],3)
        self.assertEqual(s["upstream_bbox_matched_gt"],3)
        self.assertEqual(s["upstream_known_gt_without_published_bbox_match"],1)
        self.assertEqual(s["conditional_expected_upstream_matched_prompts"],3)
        self.assertEqual(s["conditional_available_masks"],1)
        self.assertIsNone(s["bound_coverage_recall"])
        self.assertEqual(s["observed_support_recall_lower_bound_full_domain"],.25)
    def test_missing_difficult_response_cannot_improve_full_score(self):
        a=self.job("ca",[{"id":"pa","mask":self.a}],"pa")
        b=self.job("cb",[],None,"target_visibility_unknown")
        full=score_image(self.candidates,[a,b],self.targets,image_id="image")
        dropped=score_image(self.candidates,[a],self.targets,image_id="image")
        expected=[{"task_key":"a","image":{"image_id":"image"}},{"task_key":"b","image":{"image_id":"image"}}]
        args=([self.meta("image")],[],None,{"image":upstream_assignment(self.candidates,self.targets)},expected)
        sf=summarize(args[0],args[1],[full],args[3],args[4],{"a","b"})["all"]
        sd=summarize(args[0],args[1],[dropped],args[3],args[4],{"a"})["all"]
        self.assertEqual(sf["bound_strict_support_recall"],.5)
        self.assertIsNone(sd["bound_strict_support_recall"])
        self.assertEqual(sd["observed_support_recall_lower_bound_full_domain"],.5)
        self.assertEqual(sd["conditional_expected_upstream_matched_prompts"],2)
    def test_bad_mask_shape_and_unregistered_binding_reject(self):
        with self.assertRaises(ValueError):score_mask_set([{"id":"p","mask":np.zeros((2,2),bool)}],self.targets)
        with self.assertRaises(ValueError):score_mask_set([{"id":"p","mask":self.a.astype(float)}],self.targets)
        with self.assertRaises(ValueError):score_image(self.candidates,[self.job("ca",[{"id":"p","mask":self.a}],"unknown")],self.targets,image_id="image")
    def test_inherited_CPU_affinity_validated_without_reset(self):
        with patch("os.sched_setaffinity",side_effect=AssertionError("scorer must not change placement")) as setter:
            with patch("os.sched_getaffinity",return_value={47,48}):
                receipt=cpu_affinity_receipt()
                self.assertEqual(receipt["inherited_logical_CPUs"],[47,48])
                self.assertFalse(receipt["affinity_modified"])
            with patch("os.sched_getaffinity",return_value={45,46,47}):
                with self.assertRaisesRegex(ValueError,"at most2"):cpu_affinity_receipt()
            with patch("os.sched_getaffinity",return_value={0,1}):
                with self.assertRaisesRegex(ValueError,"outside"):cpu_affinity_receipt()
            setter.assert_not_called()

    def test_v1_denominators_stay_strict(self):
        cfg={"schema":"action4.sam.single_image_native.v1","upstream_role":"frozen_B0_parent_step0_post_soft_nms"}
        lock={"schema":cfg["schema"],"config":cfg};prompts=[{"image_id":str(i),"candidates":[{},{}]} for i in range(650)]
        self.assertEqual(validate_native_domain(lock,prompts,[{}]*1298)["expected_prompt_jobs"],1298)
        with self.assertRaises(ValueError):validate_native_domain(lock,prompts,[{}]*1300)
    def test_v2_source_identity_eval_only_and_step(self):
        import copy
        cfg={"schema":"action4.sam.single_image_native.v2","upstream_role":"frozen_completed_B0_post_soft_nms","split":"eval","expected_images":650,"expected_saved_candidates":1300,"expected_prompts":1300,
            "source_contract":{k:{"path":k,"sha256":"a"*64} for k in ("sealed_role","run_spec","b0_bundle","predictions")}}
        cfg["source_contract"].update(selected_step=9500,config_id="config")
        lock={"schema":cfg["schema"],"config":cfg};prompts=[{"image_id":str(i),"candidates":[{},{}]} for i in range(650)]
        seal={"role":"B0","selected_step":9500,"sha256":"a"*64,"config_id":"config","strict_load":True,"bundle":"b0_bundle","run_spec_sha256":"a"*64}
        with patch("eval.action4_sam_image_metrics.sha_file",return_value="a"*64),patch("eval.action4_sam_image_metrics.read_json",return_value=seal):
            self.assertEqual(validate_native_domain(lock,prompts,[{}]*1300)["expected_prompt_jobs"],1300)
            train=copy.deepcopy(lock);train["config"]["split"]="train"
            with self.assertRaises(ValueError):validate_native_domain(train,prompts,[{}]*1300)
            bad=copy.deepcopy(lock);bad["config"]["source_contract"]["selected_step"]=0
            with self.assertRaises(ValueError):validate_native_domain(bad,prompts,[{}]*1300)
        with patch("eval.action4_sam_image_metrics.sha_file",return_value="b"*64):
            with self.assertRaises(ValueError):validate_native_domain(lock,prompts,[{}]*1300)
    def test_paired_domain_shift_cannot_fake_mask_improvement(self):
        def row(g,iou,available=True):return {"gt_id":g,"eligible":True,"mask_available":available,"response_completed":available,"iou":iou}
        old=[{"image_id":"i","conditional":[row("A",.8),row("B",.1),row("D",.4)]}]
        new=[{"image_id":"i","conditional":[row("A",.7),row("C",1.),row("D",None,False)]}]
        domain={("i",g) for g in "ABCD"};r=paired_conditional(old,new,domain)
        self.assertEqual(r["common_conditional_eligible"],2);self.assertEqual(r["both_available_pairs"],1)
        self.assertAlmostEqual(r["mean_paired_delta_IoU"],-.1);self.assertEqual(r["common_not_both_available"],1)
        self.assertEqual(r["old_only_conditional_keys"],[["i","B"]]);self.assertEqual(r["new_only_conditional_keys"],[["i","C"]])
        with self.assertRaises(ValueError):paired_conditional(old+old,new,domain)

    def test_wholly_pending_image_retains_common_conditional_domain(self):
        candidates=self.candidates
        cohort=[self.meta("pending")]
        assignments={"pending":upstream_assignment(candidates,self.targets)}
        old=score_image(candidates,[self.job("ca",[{"id":"pa","mask":self.a}],"pa"),
            self.job("cb",[{"id":"pb","mask":self.b}],"pb")],self.targets,image_id="pending")
        scored=[]
        pending=full_conditional_domain(cohort,assignments,scored)
        self.assertEqual(scored,[])
        result=paired_conditional([old],pending,{("pending","A"),("pending","B")})
        self.assertEqual(result["common_conditional_eligible"],2)
        self.assertEqual(result["new_conditional_eligible"],2)
        self.assertEqual(result["both_available_pairs"],0)
        self.assertEqual(result["common_not_both_available"],2)
        self.assertEqual(result["old_only_conditional_keys"],[])
        self.assertTrue(all(not r["response_completed"] and r["native_status"]=="not_in_completed_score_snapshot" for r in pending[0]["conditional"]))
        complete=full_conditional_domain(cohort,assignments,[old])
        self.assertIs(complete[0],old)

    def test_completed_marker_change_rejected_before_loading_data(self):
        folder=PROJECT_ROOT/"outputs/action4_sam_selectedB0_metrics_preparation_v001/cpu_fixtures"/uuid.uuid4().hex
        folder.mkdir(parents=True);marker=folder/"COMPLETED.json";marker.write_text("{}")
        with self.assertRaisesRegex(ValueError,"changed after score snapshot"):
            load_completed_job(marker,{}, {},{},"0"*64)

if __name__=="__main__":unittest.main()
