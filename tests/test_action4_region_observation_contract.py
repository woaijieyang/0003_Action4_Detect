"""CPU evidence-contract fixtures; positive review fixture is NOT real certification."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from eval import action4_part_region_sequence as module

class RegionObservationContract(unittest.TestCase):
 def setUp(self):
  self.root=module.ROOT
  self.job=json.loads((self.root/"outputs/action4_front_region_partial_v002/JOBS.json").read_text())[0]
  with np.load(self.root/self.job["input"]["path"]) as z:
   self.parts=[dict(part_id="body",mask=z["body"].copy(),vertices_m=z["vertices"].copy(),partial_boundary=True,observed_boundary_eligibility=z["body_edge_eligibility"].copy()),dict(part_id=self.job["observed_part_role"],mask=z["part_mask"].copy(),vertices_m=z["part_vertices"].copy())]
  self.temp=tempfile.TemporaryDirectory(dir=self.root/"outputs/action4_front_region_rootcause_v001",prefix="CONTRACT_FIXTURE_")
 def tearDown(self):self.temp.cleanup()
 def reviewed(self):
  rows=[dict(part_id=p["part_id"],identity=module.region_observation_identity(self.job,p),semantic_correspondence_verified=True,physical_boundary_verified=True,full_visibility_verified=True) for p in self.parts]
  evidence=Path(self.temp.name)/"INDEPENDENT_OBSERVATION_FIXTURE.txt"
  evidence.write_text("ARTIFICIAL TEST EVIDENCE ONLY; NOT REAL GEOMETRY CERTIFICATION")
  doc=dict(schema="action4.region_observation_review.v1",test_scope="ARTIFICIAL_POSITIVE_REVIEW_NOT_REAL_CERTIFICATION",source_evidence=[module.binding(evidence)],parts=rows)
  return doc
 def bind(self,doc):
  p=Path(self.temp.name)/"review.json";p.write_text(json.dumps(doc));self.job["region_observation_review"]=module.binding(p)
 def test_seed_false_and_job_true_cannot_upgrade(self):
  seed=module.read_binding(self.job["part_receipt"]["seed_ref"])
  self.assertIs(seed["semantic_face_verified"],False)
  for value in [False,True]:
   self.job["physical_boundary_calibrated"]=value
   self.job["full_visibility_seed_review"]=True
   allowed,decision=module.resolve_region_observations(self.job,self.parts)
   self.assertEqual(len(allowed),2)
   self.assertEqual(decision["parts"][0]["status"],"research_partial_available")
   self.assertEqual(decision["parts"][1]["status"],"research_partial_unverified_semantics")
   self.assertFalse(decision["formal_decision_allowed"])
   self.assertTrue(all(p["full_visibility_seed_review"] is False for p in allowed))
 def test_exact_source_bound_explicit_review_consumable(self):
  self.bind(self.reviewed());allowed,decision=module.resolve_region_observations(self.job,self.parts)
  self.assertEqual(len(allowed),2)
  self.assertTrue(all(p["full_visibility_seed_review"] for p in allowed))
  self.assertFalse(decision["formal_decision_allowed"])
 def test_review_identity_cannot_copy_between_body_or_frame(self):
  for key,value in [("track_id","track_01"),("frame_id",809),("pts",323600)]:
   doc=self.reviewed();doc["parts"][1]["identity"][key]=value;self.bind(doc)
   with self.subTest(key=key),self.assertRaises(ValueError):module.resolve_region_observations(self.job,self.parts)
 def test_review_part_identity_cannot_copy(self):
  doc=self.reviewed();doc["parts"][1]["identity"]=copy.deepcopy(doc["parts"][0]["identity"]);self.bind(doc)
  with self.assertRaises(ValueError):module.resolve_region_observations(self.job,self.parts)
 def test_duplicate_masks_cannot_manufacture_two_cues(self):
  self.parts[1]["mask"]=self.parts[0]["mask"].copy()
  with self.assertRaises(ValueError):module.resolve_region_observations(self.job,self.parts)
 def test_missing_source_or_false_boundary_not_certified(self):
  doc=self.reviewed();doc["parts"][1]["physical_boundary_verified"]=False;self.bind(doc)
  allowed,d=module.resolve_region_observations(self.job,self.parts);self.assertEqual(len(allowed),2)
  doc=self.reviewed();doc["source_evidence"]=[];self.bind(doc)
  with self.assertRaises(ValueError):module.resolve_region_observations(self.job,self.parts)
 def test_seed_provenance_cannot_be_independent_geometry_review(self):
  doc=self.reviewed();doc["source_evidence"]=[self.job["part_receipt"]["seed_ref"]];self.bind(doc)
  with self.assertRaises(ValueError):module.resolve_region_observations(self.job,self.parts)
 def test_review_hash_change_rejected(self):
  self.bind(self.reviewed());self.job["region_observation_review"]["sha256"]="0"*64
  with self.assertRaises(ValueError):module.resolve_region_observations(self.job,self.parts)
 def test_real808_research_inputs_never_claim_full_boundary(self):
  observed=[]
  def stop(K,R,t,parts,**kwargs):
   observed.extend(parts)
   raise RuntimeError("test_capture_before_solver")
  with patch.object(module,"refine_clear_seed_regions",side_effect=stop):
   result=module.fit_track(([self.job],24))[0]
  self.assertEqual(len(observed),2)
  self.assertTrue(all(p["partial_boundary"] and not p["full_visibility_seed_review"] for p in observed))
  self.assertFalse(result["published"]);self.assertFalse(result["formal_decision_allowed"])
  self.assertEqual(result["final_corners"],self.job["A0_raw"]["corners_raw_from_resolved_roi"])
