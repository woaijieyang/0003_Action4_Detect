import copy,json,hashlib,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from data.action4_paths import PROJECT_ROOT as R
from model.training.action4_corner_research_selection import candidate,commit,verify_complete,rank,POLICY
from model.training.action4_roi115_finetune import validate_parent

class ResearchSelectionTests(unittest.TestCase):
 def setUp(self):
  self.root=Path(tempfile.mkdtemp(prefix="research_selection_",dir=R/"outputs/action4_roi115_finetune_preparation_v001"))
 def ref(self,name,value):
  p=self.root/name;p.write_text(json.dumps(value));return dict(path=str(p.relative_to(R)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 def result(self,step,success,ready=False):
  rows=[dict(sample_id="image"+str(i//2),instance_id=i%2,all8_at_002=i<success,all8_at_005=i<success) for i in range(1300)]
  m=dict(score_schema="action4.pose_projection_all8_002.v1",pose_gt_count=1300,pose_output_count=1300,per_gt_failure_reason=rows,
   pose_all8_at_002_success=success,pose_all8_at_002_rate=success/1300,pose_fp=1300-success,pose_fn=1300-success,pose_corner_p95_valid_matched_only=1.)
  return dict(config_id="fixture_only_not_model_result",step=step,research_selection_policy=POLICY,deployment_eligible=False,
   evaluated_snapshot=self.ref("snapshot"+str(step)+".pt",dict(fixture_step=step)),
   report=dict(candidate_pose_report=m,evaluation_images=650,corner_quality_ready=ready,
    candidate_pose_report_artifact=self.ref("candidate"+str(step)+".json",m),calibration_receipt=self.ref("calibration"+str(step)+".json",dict(quality_ready=ready))))
 def test_quality_false_does_not_block_candidate_and_atomic_alias(self):
  a=self.result(0,100);b=self.result(10,150);c=self.result(20,120)
  commit(self.root,a);commit(self.root,b);commit(self.root,c)
  got=verify_complete(self.root,[c,a,b]);self.assertEqual(got["step"],10);self.assertFalse(got["quality_ready"])
  self.assertTrue(os.path.samefile(self.root/"pnp_research_best.pt",R/b["evaluated_snapshot"]["path"]))
  self.assertFalse((self.root/"BEST.json").exists())
 def test_duplicate_incomplete_domain_rejected(self):
  a=self.result(0,100);a["report"]["candidate_pose_report"]["per_gt_failure_reason"].pop()
  with self.assertRaises(ValueError):candidate(a)
 def test_tie_earlier_step_and_wrong_run_rejected(self):
  a=self.result(0,100);b=self.result(10,100)
  self.assertGreater(rank(candidate(a)),rank(candidate(b)))
  commit(self.root,a);b["config_id"]="wrong"
  with self.assertRaises(ValueError):commit(self.root,b)
 def test_controller_rejects_wrong_best(self):
  a=self.result(0,100);b=self.result(10,150);commit(self.root,a)
  with self.assertRaises(ValueError):verify_complete(self.root,[a,b])
 def test_parent_status_and_selected_route_must_be_exact(self):
  cfg=dict(roi_finetune=dict(schema="action4.corner_roi115_finetune.v1",parent_roi_expansion=1.5,roi_expansion=1.15,epochs=75),
    training_protocol="action4.corner_prediction_roi_enhancement.native_v1",stage4_parent_seal={},parent_bundle_sha256="a"*64,parent_best_step=7300)
  seal=dict(schema="action4.sealed_corner_all8_research.v1",status="SEALED_USER_STOPPED_ALL8_RESEARCH",selected_route="corner_candidate",
   strict_load=True,frozen_backbone_verified=True,sha256="a"*64,selected_step=7300,roi_expansion=1.5)
  for field,value in [("status","unsealed"),("selected_route","raw")]:
   wrong=dict(seal,**{field:value})
   with patch("model.training.action4_roi115_finetune._v2_checked",return_value=wrong):
    with self.assertRaises(ValueError):validate_parent(cfg)

if __name__=="__main__":unittest.main()
