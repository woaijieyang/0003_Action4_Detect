"""CPU metadata fixtures for BPQ1; not actual visual approval or optimizer runs."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest import mock
from data import action4_body_projection_fit as fit
BASE=fit.ROOT/"outputs/action4_realism_reassessment_v001/body_projection_fit_v001/forward_review_binding_fix_v001"
class ForwardReviewBindingTests(unittest.TestCase):
 def setUp(self):
  self.case=Path(tempfile.mkdtemp(prefix="cpu_fixture_",dir=BASE))
  self.forward=self.case/"forward";self.forward.mkdir()
  for name in fit.FORWARD_ARTIFACTS:
   (self.forward/name).write_text(json.dumps({"scope":"CPU hash fixture, not rendered or reviewed image","name":name}))
  self.manifest=fit.write_forward_manifest(self.forward,fit.forward_sources())
  self.review={"schema_version":"action4.body_forward_review.v1","actually_viewed":True,
   "decision":"proceed_bounded_body_only_diagnostic_not_alignment_acceptance",
   "reviewer":"CPU_CONTRACT_FIXTURE_NOT_ACTUAL_VISUAL_REVIEW","observations":"Synthetic metadata fixture only",
   "forward_manifest":{"path":str((self.forward/"forward_manifest.json").relative_to(fit.ROOT)),"sha256":fit.sha(self.forward/"forward_manifest.json")},
   "reviewed_artifacts":{n:self.manifest["artifacts"][n] for n in fit.REVIEW_IMAGES}}
  self.save_review()
 def save_review(self):
  (self.forward/"review.json").write_text(json.dumps(self.review,indent=2))
 def test_valid_exact_binding(self):
  r=fit.validate_forward_review(self.forward)
  self.assertEqual(r["reviewed_artifacts"],self.review["reviewed_artifacts"])
 def test_legacy_boolean_receipt_rejected(self):
  (self.forward/"review.json").write_text('{"actually_viewed":true}')
  with self.assertRaisesRegex(ValueError,"Explicit"):fit.validate_forward_review(self.forward)
 def test_actual_source_fixture_bytes_changed(self):
  extra=self.case/"source_fixture.txt";extra.write_text("before")
  sources=fit.FORWARD_SOURCES+(str(extra.relative_to(fit.ROOT)),)
  with mock.patch.object(fit,"FORWARD_SOURCES",sources):
   manifest=copy.deepcopy(self.manifest);manifest["sources"]=fit.forward_sources()
   (self.forward/"forward_manifest.json").write_text(json.dumps(manifest))
   self.review["forward_manifest"]["sha256"]=fit.sha(self.forward/"forward_manifest.json");self.save_review()
   extra.write_text("after")
   with self.assertRaisesRegex(ValueError,"Stale forward source"):fit.validate_forward_review(self.forward)
 def test_stale_image_rejected(self):
  (self.forward/fit.REVIEW_IMAGES[0]).write_text("changed image")
  with self.assertRaisesRegex(ValueError,"Stale forward artifact"):fit.validate_forward_review(self.forward)
 def test_stale_feature_definitions_rejected(self):
  (self.forward/"feature_definitions.json").write_text('{"different_features":true}')
  with self.assertRaisesRegex(ValueError,"feature_definitions"):fit.validate_forward_review(self.forward)
 def test_stale_forward_report_rejected(self):
  (self.forward/"forward_report.json").write_text('{"different_forward":true}')
  with self.assertRaisesRegex(ValueError,"forward_report"):fit.validate_forward_review(self.forward)
 def test_manifest_bytes_changed_rejected_even_same_json(self):
  p=self.forward/"forward_manifest.json";p.write_text(p.read_text()+"\n")
  with self.assertRaisesRegex(ValueError,"exact forward manifest"):fit.validate_forward_review(self.forward)
 def test_review_wrong_image_binding_rejected(self):
  self.review["reviewed_artifacts"][fit.REVIEW_IMAGES[0]]="0"*64;self.save_review()
  with self.assertRaisesRegex(ValueError,"four exact"):fit.validate_forward_review(self.forward)
 def test_unrelated_manifest_path_rejected(self):
  self.review["forward_manifest"]["path"]="outputs/unrelated/forward_manifest.json";self.save_review()
  with self.assertRaisesRegex(ValueError,"exact forward manifest"):fit.validate_forward_review(self.forward)
 def test_missing_artifact_coverage_rejected(self):
  manifest=copy.deepcopy(self.manifest);manifest["artifacts"].pop("feature_definitions.json")
  (self.forward/"forward_manifest.json").write_text(json.dumps(manifest))
  with self.assertRaisesRegex(ValueError,"coverage"):fit.validate_forward_review(self.forward)
 def test_reject_before_fit_or_output_creation(self):
  self.review["actually_viewed"]=False;self.save_review();out=self.case/"should_not_exist"
  with mock.patch.object(fit,"prepare") as prepare,mock.patch.object(fit,"least_squares") as optimizer:
   with self.assertRaises(ValueError):fit.run(out,self.forward)
   prepare.assert_not_called();optimizer.assert_not_called()
  self.assertFalse(out.exists())
 def test_source_changed_during_forward_generation_rejects(self):
  old=fit.forward_sources();old[fit.PACKET]="0"*64
  with self.assertRaisesRegex(ValueError,"during forward"):fit.write_forward_manifest(self.case/"unused",old)
 def test_symlink_cannot_escape_forward_directory(self):
  target=self.case/"external_image";target.write_text("outside forward")
  image=self.forward/fit.REVIEW_IMAGES[0];image.unlink();image.symlink_to(target)
  with self.assertRaisesRegex(ValueError,"inside their outputs"):fit.validate_forward_review(self.forward)
if __name__=="__main__":unittest.main(verbosity=2)
