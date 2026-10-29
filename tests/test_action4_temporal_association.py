"""Prediction-only association contracts, not native SAM tracking tests."""
import unittest
import numpy as np
from model.action4_guidance_contract import FrameIdentity,SourceIdentity
from model.action4_temporal_association import TrackEvidence,CandidateEvidence,associate
class AssociationTests(unittest.TestCase):
 def setUp(self):
  self.frame=FrameIdentity("frame20",20,"native",64,64,"a"*64)
  self.a=tuple(np.eye(128)[0]);self.b=tuple(np.eye(128)[1])
  self.source=SourceIdentity("b0","b"*64,"c"*64)
 def track(self,tid="right",box=(30,30,40,40),appearance=None,**kw):
  args=dict(frame=self.frame,track_id=tid,generation=0,support_box=box,appearance128=self.a if appearance is None else appearance,
    diagonal_px=15.,last_observed_frame=19,last_observed_seconds=.63,support_origin="history_prediction",
    support_id="history:"+tid,identity_confirmed=True,appearance_source=self.source)
  args.update(kw);return TrackEvidence(**args)
 def candidate(self,cid,box,score=.5,appearance=None,**kw):
  args=dict(frame=self.frame,candidate_id=cid,support_box=box,appearance128=self.a if appearance is None else appearance,
    detector_score=score,association_eligible=True,support_id="candidate:"+cid,appearance_source=self.source)
  args.update(kw);return CandidateEvidence(**args)
 def test_history_can_select_lower_detector_score(self):
  track=self.track()
  shirt=self.candidate("shirt",(5,5,20,20),.95)
  camera=self.candidate("camera",(31,30,41,40),.6)
  result=associate(self.frame,.66,[track],[shirt,camera])
  self.assertEqual(result["matches"][0]["candidate_id"],"camera")
  self.assertEqual(result["matches"][0]["match_basis"],"local_box")
  self.assertFalse(result["A0_ROI_publication_ready"])
  self.assertTrue(any(e.get("rejected")=="motion_gate" for e in result["edge_diagnostics"]))
 def test_overlap_allowed_and_permutation_deterministic(self):
  tracks=[self.track("A",(10,10,30,30),self.a,diagonal_px=30),
          self.track("B",(18,10,38,30),self.b,diagonal_px=30)]
  candidates=[self.candidate("a",(10,10,30,30),appearance=self.a),
              self.candidate("b",(18,10,38,30),appearance=self.b)]
  a=associate(self.frame,.66,tracks,candidates);b=associate(self.frame,.66,tracks[::-1],candidates[::-1])
  self.assertEqual(a,b);self.assertEqual(len(a["matches"]),2)
  self.assertEqual(len({x["candidate_id"] for x in a["matches"]}),2)
 def test_similar_cost_does_not_follow_highest_probability(self):
  cs=[self.candidate("a",(30,30,40,40),.99),self.candidate("b",(30.1,30,40.1,40),.4)]
  result=associate(self.frame,.66,[self.track()],cs)
  self.assertEqual(result["matches"],[])
  self.assertEqual(result["unmatched_tracks"][0]["reason"],"ambiguous_candidate_margin")
 def test_expiry_and_unknown_identity_are_explicit(self):
  for kw,reason in [({"identity_confirmed":False},"unconfirmed_identity"),
       ({"last_observed_frame":5},"lost"),({"last_observed_seconds":.1},"lost"),
       ({"appearance128":None},"appearance_unavailable")]:
   result=associate(self.frame,.66,[self.track(**kw)],[self.candidate("a",(30,30,40,40))])
   self.assertEqual(result["matches"],[]);self.assertEqual(result["unmatched_tracks"][0]["reason"],reason)
 def test_frame_and_id_mismatch_rejected(self):
  wrong=FrameIdentity("other",20,"native",64,64,"b"*64)
  with self.assertRaises(ValueError):associate(self.frame,.66,[self.track(frame=wrong)],[])
  with self.assertRaises(ValueError):associate(self.frame,.66,[self.track(),self.track()],[])
  with self.assertRaises(ValueError):self.track(support_origin="GT")
 def test_shared_masks_are_not_two_independent_objects(self):
  mask=np.zeros((64,64),bool);mask[10:40,10:40]=True
  ts=[self.track("A",support_origin="sam_prediction",predicted_mask=mask),
      self.track("B",support_origin="sam_prediction",predicted_mask=mask.copy())]
  result=associate(self.frame,.66,ts,[self.candidate("a",(30,30,40,40))])
  self.assertFalse(result["matches"])
  self.assertTrue(all(x["reason"]=="ambiguous_shared_mask" for x in result["unmatched_tracks"]))
  cs=[self.candidate("a",(30,30,40,40),predicted_mask=mask),
      self.candidate("b",(5,5,10,10),predicted_mask=mask.copy())]
  result=associate(self.frame,.66,[self.track()],cs)
  self.assertEqual(result["ambiguous_candidates"],["a","b"]);self.assertFalse(result["matches"])
 def test_different_model_embeddings_cannot_be_compared(self):
  candidate=self.candidate("camera",(30,30,40,40),appearance_source=SourceIdentity("b0","d"*64,"c"*64))
  result=associate(self.frame,.66,(t for t in [self.track()]),(c for c in [candidate]))
  self.assertEqual(result["matches"],[])
  self.assertEqual(result["edge_diagnostics"][0]["rejected"],"appearance_source_mismatch")
  self.assertEqual(result["frame_identity"]["rgb_sha256"],self.frame.rgb_sha256)
 def test_actual_masks_have_distinct_basis(self):
  mask=np.zeros((64,64),bool);mask[30:40,30:40]=True
  t=self.track(support_origin="sam_prediction",predicted_mask=mask)
  c=self.candidate("camera",(30,30,40,40),predicted_mask=mask)
  result=associate(self.frame,.66,[t],[c])
  self.assertEqual(result["matches"][0]["match_basis"],"mask")
  with self.assertRaises(ValueError):t.predicted_mask.setflags(write=True)
if __name__=="__main__":unittest.main(verbosity=2)
