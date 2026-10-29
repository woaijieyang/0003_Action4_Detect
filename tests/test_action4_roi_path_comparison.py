import copy,unittest
from eval.action4_roi_path_comparison import compare
class Paired(unittest.TestCase):
 def fixture(self):
  identity={k:"a"*64 for k in ["A0_sha256","B0_sha256","code_sha256","camera_identity","canonical8_identity","decoder_identity","pixel_space_id"]}
  identity["roi_expansion"]=1.5
  identity.update(temporal_mode="causal_forward",causal=True)
  manifest=[dict(frame_id=7,pts=2800,RGB_sha256="b"*64)]
  p=dict(grid_index=5,bbox_full_xyxy=[0,0,10,10],pose_valid=True,corner_ids=list(range(8)),corners_px=[[1,2]]*8)
  t=dict(p,track_id="t0",bbox_for_a0=[1,0,11,10],B0_current_observation=True,temporal_status="observed",
      corners_raw_from_resolved_roi=[[2,2]]*8,corners_final=[[2,2]]*8,post_A0_geometry="none")
  row=dict(manifest[0],roi_expansion=1.5,prediction={"instances":[p]})
  other=dict(manifest[0],roi_expansion=1.5,prediction={"instances":[t]})
  return [row],[other],dict(b0_identity=identity,track_identity=copy.deepcopy(identity),frame_manifest=manifest)
 def test_common_outputs_not_called_accuracy(self):
  a,b,k=self.fixture();r=compare(a,b,**k)
  self.assertEqual(r["counts"]["common_b0_candidate"],1);self.assertEqual(r["common_corner_delta"]["median"],1)
  self.assertEqual(r["counts"]["review_unknown"],1);self.assertFalse(r["accuracy_GT_available"])
 def test_recovered_missing_and_review_are_separate(self):
  a,b,k=self.fixture();a[0]["prediction"]["instances"]=[]
  t=b[0]["prediction"]["instances"][0];t.update(grid_index=None,B0_current_observation=False,temporal_status="sam_supported")
  r=compare(a,b,**k);self.assertEqual(r["counts"]["track_supported_without_current_b0"],1);self.assertIsNone(r["recovered_correct_instances"])
  r=compare(a,b,review={"7:t0":"incorrect"},**k)
  self.assertEqual(r["counts"]["reviewed_incorrect"],1);self.assertEqual(r["recovered_correct_instances"],0)
 def test_missing_frame_or_wrong_model_rejected(self):
  a,b,k=self.fixture()
  with self.assertRaises(ValueError):compare(a,[],**k)
  k["track_identity"]["A0_sha256"]="c"*64
  with self.assertRaises(ValueError):compare(a,b,**k)
 def test_duplicate_instance_and_smoothing_rejected(self):
  a,b,k=self.fixture();b[0]["prediction"]["instances"].append(copy.deepcopy(b[0]["prediction"]["instances"][0]))
  b[0]["prediction"]["instances"][1]["track_id"]="t1"
  with self.assertRaises(ValueError):compare(a,b,**k)
  a,b,k=self.fixture();b[0]["prediction"]["instances"][0]["corners_final"]=[[3,2]]*8
  with self.assertRaises(ValueError):compare(a,b,**k)
 def test_unadopted_B0_not_hidden(self):
  a,b,k=self.fixture();b[0]["prediction"]["instances"]=[]
  r=compare(a,b,**k);self.assertEqual(r["counts"]["b0_only_not_adopted"],1)
  self.assertIsNone(r["common_corner_delta"]["median"])

 def test_corner_permutation_rejected(self):
  a,b,k=self.fixture();b[0]["prediction"]["instances"][0]["corner_ids"]=list(reversed(range(8)))
  with self.assertRaises(ValueError):compare(a,b,**k)

 def test_bidirectional_cannot_enter_causal_comparison(self):
  a,b,k=self.fixture();k["track_identity"].update(temporal_mode="offline_bidirectional_clip",causal=False)
  with self.assertRaises(ValueError):compare(a,b,**k)
  r=compare(a,b,comparison_mode="offline_bidirectional_SAM_diagnostic",**k)
  self.assertEqual(r["comparison_mode"],"offline_bidirectional_SAM_diagnostic")
