"""Pure calibration arithmetic fixtures, never model/effect admission evidence."""
import copy,unittest
from eval.action4_corner_calibration import calibrate,digest,RUN_KEYS

class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.run={k:("a" if k!="checkpoint_sha256" else "b")*64 for k in RUN_KEYS};self.run["identity"]=digest(self.run)
        images=[f"image{i:03d}" for i in range(650)]
        units=[{"image_id":images[i],"gt_instance_id":"1","eligible":True,"pose_nonignore":True,"pose_groups":["all","small" if i%2 else "occluded"],"groups":["all","small" if i%2 else "occluded"]} for i in range(130)]
        self.records=[];point=0;predictions=[]
        for i,u in enumerate(units):
            row={"image_id":u["image_id"],"gt_instance_id":"1","prediction_id":f"prediction{i}" if i<13 else None,"run_identity":self.run["identity"]}
            if i<13:
                count=8 if i<11 else 6
                selected=[j<count for j in range(8)]
                errors=[]
                for j in range(8):
                    errors.append(1. if point<90 else 1.001)
                    if selected[j]:point+=1
                row.update(corner_ids=list(range(8)),q_corner=[.6 if v else .1 for v in selected],p_valid=[.8]*8,
                    uv224=[[100.,100.]]*8,multimodal=[False]*8,label_known=[True]*8,gt_valid=[True]*8,
                    error224=errors,gt_diagonal224=100.)
                predictions.append({"image_id":u["image_id"],"prediction_id":row["prediction_id"]})
            self.records.append(row)
        self.domain={"schema":"action4.corner_calibration_domain.v1","split":"eval","image_ids":images,"gt_units":units,
            "prediction_units":predictions,"data_sha256":self.run["data_sha256"],"predictions_sha256":self.run["predictions_sha256"]}
        self.domain["domain_sha256"]=digest(self.domain)
        self.raw={"run_identity":self.run["identity"],"data_sha256":self.run["data_sha256"],
            "score_schema":"action4.pose_projection_all8_002.v1","complete":True,"image_ids":images,
            "pose_gt_count":130,"pose_all8_at_002_success":10,
            "outcomes":[{"image_id":u["image_id"],"gt_instance_id":"1","success":i<10} for i,u in enumerate(units)],
            "metrics":{"other_statistics_retained":"fixture"}}
        self.candidate=copy.deepcopy(self.raw)
    def sync_report(self,report):
        report["schema"]="action4.corner_calibration_pose_outcomes.v1"
        if any(row["success"] is None for row in report["outcomes"]):
            report["complete"]=False;report["pose_report"]=None;return
        n=report["pose_gt_count"];tp=report["pose_all8_at_002_success"]
        predictions={(r["image_id"],r["prediction_id"]) for r in self.domain["prediction_units"]}
        matched=set();rows=[]
        for row in report["outcomes"]:
            pid=next((p for image,p in sorted(predictions) if image==row["image_id"]),None) if row["success"] else None
            if pid is not None:matched.add((row["image_id"],pid))
            rows.append({"sample_id":row["image_id"],"instance_id":row["gt_instance_id"],
                "prediction_id":pid,"errors_px":[.5]*8 if pid is not None else None,"all8_at_002":row["success"],"all8_at_005":row["success"],
                "failure_reason":None if row["success"] else "no_valid_pose_match"})
        count=len(predictions);ratio=lambda a,b:a/b if b else None
        pose_units={(u["image_id"],str(u["gt_instance_id"])):u for u in self.domain["gt_units"] if u["pose_nonignore"]}
        groups={}
        for row in rows:
            for name in pose_units[(row["sample_id"],str(row["instance_id"]))]["pose_groups"]:
                group=groups.setdefault(name,{"gt":0,"tp002":0});group["gt"]+=1;group["tp002"]+=row["all8_at_002"]
        for group in groups.values():group["recall"]=ratio(group["tp002"],group["gt"])
        strict=sum(sum(r["all8_at_002"] for r in rows if r["sample_id"]==image)==sum(r["sample_id"]==image for r in rows)
            and sum(i==image for i,p in predictions)==sum(r["sample_id"]==image for r in rows) for image in self.domain["image_ids"])
        report["pose_report"]={"score_schema":"action4.pose_projection_all8_002.v1","pose_gt_count":n,"pose_output_count":count,
            "pose_all8_at_002_success":tp,"pose_all8_at_002_rate":ratio(tp,n),"pose_all8_at_005_rate":ratio(tp,n),
            "pose_all8_at_002_precision":ratio(tp,count),"pose_all8_at_002_f1":ratio(2*tp,n+count),
            "pose_fp":count-tp,"pose_fn":n-tp,"invalid_pose_count":0,"pose_instance_precision":ratio(len(matched),count),
            "pose_corner_p95_valid_matched_only":.5 if matched else None,"strict_image_success_rate":ratio(strict,len(self.domain["image_ids"])),
            "per_gt_failure_reason":rows,"unmatched_pose_predictions":[{"sample_id":i,"prediction_id":p,"reason":"unmatched_pose"} for i,p in sorted(predictions-matched)],
            "pose_groups":groups}
    def call(self):
        self.sync_report(self.raw);self.sync_report(self.candidate)
        return calibrate(self.domain,self.run,self.records,self.raw,self.candidate)
    def test_exact100_90percent_10percent_and_global_choice(self):
        r=self.call();s=r["point_and_instance_statistics"]
        self.assertEqual(s["selected_valid_points"],100);self.assertEqual(s["selected_known_points"],100)
        self.assertEqual(s["selected_correct_points"],90);self.assertEqual(s["instances_with_6_usable_ids"],13)
        self.assertEqual(s["eligible_gt_instances"],130);self.assertEqual(s["missing_prediction_instances"],117)
        self.assertTrue(r["quality_ready"]);self.assertTrue(r["quality_gate_passed_for_selection"]);self.assertEqual(r["final_path"],"candidate")
        self.assertEqual(r["candidate_report"]["metrics"],self.candidate["metrics"])
    def test_independent_boundaries_fail(self):
        self.records[0]["q_corner"][0]=.59
        self.assertIn("insufficient_GT_valid_selected_points",self.call()["reasons"])
        self.setUp();self.records[0]["error224"][0]=1.00001
        self.assertIn("selected_known_precision_below90percent",self.call()["reasons"])
        self.setUp();self.records[12]["p_valid"][0]=.79
        self.assertIn("six_usable_ID_instance_coverage_below10percent",self.call()["reasons"])
        self.setUp();self.candidate["outcomes"][0]["success"]=False;self.candidate["pose_all8_at_002_success"]=9
        r=self.call();self.assertIn("candidate_all8_below_raw",r["reasons"]);self.assertEqual(r["final_report"],r["raw_report"])
    def test_known_invalid_selected_are_precision_negatives(self):
        self.records[12]["q_corner"][6:]=[.6,.6];self.records[12]["gt_valid"][6:]=[False,False]
        self.records[12]["uv224"][6:]=[[-1.,100.],[224.,100.]];self.records[12]["error224"][6:]=[None,None]
        s=self.call()["point_and_instance_statistics"]
        self.assertEqual(s["selected_known_points"],102);self.assertEqual(s["selected_valid_points"],100)
        self.assertEqual(s["selected_correct_points"],90);self.assertFalse(self.call()["quality_ready"])
    def test_unknown_and_noGT_are_unmeasured(self):
        self.records[0]["label_known"][0]=False;self.records[0]["error224"][0]=None
        r=self.call();self.assertFalse(r["quality_ready"]);self.assertIsNone(r["point_and_instance_statistics"]["selected_known_precision"])
        self.assertEqual(r["point_and_instance_statistics"]["selected_unknown_points"],1)
        self.setUp();self.records[0]["error224"][0]=None
        r=self.call();self.assertEqual(r["point_and_instance_statistics"]["unmeasured_selected_valid_points"],1)
        self.assertFalse(r["quality_ready"])
    def test_FP_preserved_not_instance_inflation_or_implicit_background(self):
        fp=copy.deepcopy(self.records[0]);fp.update(gt_instance_id=None,prediction_id="unmatched")
        fp["gt_valid"]=[False]*8;fp["error224"]=[None]*8
        self.records.append(fp);self.domain["prediction_units"].append({"image_id":fp["image_id"],"prediction_id":"unmatched"})
        self.domain["domain_sha256"]=digest({k:v for k,v in self.domain.items() if k!="domain_sha256"})
        r=self.call();s=r["point_and_instance_statistics"]
        self.assertEqual(s["selected_known_points"],108);self.assertEqual(s["instances_with_6_usable_ids"],13)
        self.assertEqual(s["eligible_gt_instances"],130);self.assertFalse(r["quality_ready"])
        fp["label_known"]=[False]*8
        r=self.call();self.assertEqual(r["point_and_instance_statistics"]["selected_unknown_points"],8)
        self.assertIsNone(r["point_and_instance_statistics"]["selected_known_precision"])
    def test_missing_coverage_not_silently_zero(self):
        self.records.pop(0)
        r=self.call();self.assertIn("incomplete_GT_record_coverage",r["reasons"]);self.assertIn("incomplete_prediction_record_coverage",r["reasons"])
        self.setUp();self.raw["image_ids"]=self.raw["image_ids"][:-1]
        self.assertIn("pose_score_domain_incomplete_or_unmeasured",self.call()["reasons"])
        self.setUp();self.candidate["outcomes"][0]["success"]=None;self.candidate["pose_all8_at_002_success"]=9
        self.assertIn("pose_score_domain_incomplete_or_unmeasured",self.call()["reasons"])
    def test_duplicates_and_mixed_binding_reject(self):
        self.records.append(copy.deepcopy(self.records[0]))
        with self.assertRaises(ValueError):self.call()
        self.setUp();self.records[1]["image_id"]=self.records[0]["image_id"];self.records[1]["gt_instance_id"]="other"
        with self.assertRaises(ValueError):self.call()
        self.setUp();self.records[0]["run_identity"]="c"*64
        with self.assertRaises(ValueError):self.call()
        self.setUp();self.candidate["score_schema"]="different"
        with self.assertRaises(ValueError):self.call()
        self.setUp();self.domain["data_sha256"]="c"*64
        with self.assertRaises(ValueError):self.call()
    def test_order_invariance_and_no_per_image_selection(self):
        a=self.call();self.records.reverse();self.raw["outcomes"].reverse();self.candidate["outcomes"].reverse()
        b=self.call();self.assertEqual(a,b)
        self.records[0]["choose_candidate_using_GT"]=True
        with self.assertRaises(ValueError):self.call()
    def test_unknown_low_q_and_duplicate_FP_cannot_evade_gate(self):
        self.records[12]["label_known"][7]=False
        r=self.call();self.assertIn("quality_labels_unmeasured",r["reasons"]);self.assertFalse(r["quality_ready"])
        self.setUp()
        fp=copy.deepcopy(self.records[0]);fp["gt_instance_id"]=None;fp["gt_valid"]=[False]*8
        self.records.append(fp)
        with self.assertRaisesRegex(ValueError,"Duplicate prediction"):self.call()

    def test_no_GT_domain_is_explicitly_unmeasured(self):
        self.domain["gt_units"]=[]
        self.domain["domain_sha256"]=digest({k:v for k,v in self.domain.items() if k!="domain_sha256"})
        self.records=[r for r in self.records if r["prediction_id"] is not None]
        for row in self.records:
            row.update(gt_instance_id=None,label_known=[False]*8,gt_valid=[False]*8,error224=[None]*8,gt_diagonal224=None)
        for report in (self.raw,self.candidate):
            report.update(pose_gt_count=0,pose_all8_at_002_success=0,outcomes=[])
        result=self.call()
        self.assertFalse(result["quality_ready"]);self.assertFalse(result["quality_gate_passed_for_selection"])
        self.assertIn("no_eligible_GT_instances",result["reasons"])
        self.assertIsNone(result["raw_statistics"]["all8_at_2_percent"])
        self.assertEqual(result["point_and_instance_statistics"]["selected_correct_points"],0)

    def test_full_pose_denominator_is_not_quality_subdomain(self):
        self.domain["gt_units"][-1]["eligible"]=False
        self.domain["domain_sha256"]=digest({k:v for k,v in self.domain.items() if k!="domain_sha256"})
        result=self.call()
        self.assertEqual(result["point_and_instance_statistics"]["eligible_gt_instances"],129)
        self.assertEqual(result["raw_statistics"]["denominator"],130)
        self.raw["pose_gt_count"]=129
        with self.assertRaisesRegex(ValueError,"all nonignore"):self.call()
    def test_original_pose_counts_and_outcomes_are_crosschecked(self):
        self.call()
        for key in ("pose_gt_count","pose_all8_at_002_success","pose_fp","pose_output_count"):
            raw=copy.deepcopy(self.raw);raw["pose_report"][key]+=1
            with self.assertRaises(ValueError):
                calibrate(self.domain,self.run,self.records,raw,self.candidate)
        raw=copy.deepcopy(self.raw);raw["pose_report"]["per_gt_failure_reason"][0]["all8_at_002"]=False
        with self.assertRaises(ValueError):calibrate(self.domain,self.run,self.records,raw,self.candidate)
        raw=copy.deepcopy(self.raw);raw["pose_report"]["unmatched_pose_predictions"].pop()
        with self.assertRaises(ValueError):calibrate(self.domain,self.run,self.records,raw,self.candidate)
    def test_reviewed_forged_pose_statistics_all_reject(self):
        self.call()
        mutations=[
            ("group_recall",lambda p:p["pose_groups"]["all"].update(recall=1.)),
            ("invalid_negative",lambda p:p.update(invalid_pose_count=-7)),
            ("p95_negative",lambda p:p.update(pose_corner_p95_valid_matched_only=-9)),
            ("005_out_of_range",lambda p:p.update(pose_all8_at_005_rate=2.)),
            ("group_GT",lambda p:p["pose_groups"]["small"].update(gt=64)),
            ("group_TP",lambda p:p["pose_groups"]["small"].update(tp002=0,recall=0.)),
            ("positive_wrong_P95",lambda p:p.update(pose_corner_p95_valid_matched_only=.6)),
            ("005_wrong_bounded",lambda p:p.update(pose_all8_at_005_rate=.9)),
            ("strict_wrong_bounded",lambda p:p.update(strict_image_success_rate=.9)),
            ("invalid_wrong_bounded",lambda p:p.update(invalid_pose_count=1)),
        ]
        for name,mutate in mutations:
            report=copy.deepcopy(self.candidate);mutate(report["pose_report"])
            with self.subTest(name=name):
                with self.assertRaises(ValueError):calibrate(self.domain,self.run,self.records,self.raw,report)
    def test_unmeasured_matched_errors_block_readiness(self):
        self.call();report=copy.deepcopy(self.candidate)
        report["pose_report"]["per_gt_failure_reason"][0]["errors_px"]=None
        result=calibrate(self.domain,self.run,self.records,self.raw,report)
        self.assertFalse(result["quality_ready"]);self.assertFalse(result["candidate_statistics"]["derived_statistics_measured"])
    def test_quality_exclusion_is_not_pose_ignore(self):
        self.domain["gt_units"][-1]["eligible"]=False
        self.domain["domain_sha256"]=digest({k:v for k,v in self.domain.items() if k!="domain_sha256"})
        stats=self.call()["point_and_instance_statistics"]
        self.assertEqual(stats["quality_ineligible_gt"],1);self.assertEqual(stats["pose_ignored_gt"],0)
        self.assertNotIn("ignored_gt_instances",stats)

    def test_quality_ready_does_not_admit_stage_best(self):
        from eval.action4_pose_protocol import eligible
        result=self.call();self.assertTrue(result["quality_ready"])
        self.assertTrue(result["quality_gate_passed_for_selection"])
        self.assertNotIn("best_eligible",result)
        self.assertEqual(result["selection_eligibility"],"not_decided_here")
        final=result["final_report"]["pose_report"];reference=copy.deepcopy(final)
        reference["pose_instance_precision"]=1.;reference["pose_all8_at_002_precision"]=1.
        self.assertFalse(eligible(final,reference))

    def test_full650_required_and_groups_overlap(self):
        r=self.call();self.assertEqual(r["groups"]["all"]["eligible_gt_instances"],130)
        self.assertEqual(r["groups"]["small"]["eligible_gt_instances"],65)
        self.domain["image_ids"]=self.domain["image_ids"][:649]
        self.domain["domain_sha256"]=digest({k:v for k,v in self.domain.items() if k!="domain_sha256"})
        self.raw["image_ids"]=self.raw["image_ids"][:649];self.candidate["image_ids"]=self.candidate["image_ids"][:649]
        self.assertIn("full650_image_domain_required",self.call()["reasons"])

if __name__=="__main__":unittest.main()
