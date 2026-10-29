import unittest
from model.action4_region_observation_gate import RegionObservationGate
class RegionGateTests(unittest.TestCase):
 def test_hysteresis_and_missing_evidence(self):
  g=RegionObservationGate("a")
  args=dict(track_id="a",b0_box=[0,0,10,10],current_B0=True,current_SAM=True,prior_available=True)
  self.assertFalse(g.observe(frame_id=1,native_box=[20,20,30,30],**args)["enabled"])
  self.assertFalse(g.observe(frame_id=2,native_box=[5,0,15,10],**args)["enabled"])
  self.assertTrue(g.observe(frame_id=3,native_box=[0,0,10,10],**args)["enabled"])
  args["prior_available"]=False
  self.assertTrue(g.observe(frame_id=4,native_box=[20,20,30,30],**args)["enabled"])
 def test_wrong_identity_or_stale_rejected(self):
  g=RegionObservationGate("a")
  args=dict(native_box=[0,0,10,10],b0_box=[0,0,10,10],current_B0=True,current_SAM=True,prior_available=True)
  with self.assertRaises(ValueError):g.observe(track_id="b",frame_id=1,**args)
  g.observe(track_id="a",frame_id=1,**args)
  with self.assertRaises(ValueError):g.observe(track_id="a",frame_id=1,**args)

 def test_resolved_ROI_continues_without_current_B0(self):
  g=RegionObservationGate("a")
  args=dict(track_id="a",b0_box=None,current_B0=False,current_SAM=True,prior_available=True,
   resolved_box=[0,0,10,10],resolved_current_support=True)
  r=g.observe(frame_id=1,native_box=[20,20,30,30],**args)
  self.assertFalse(r["enabled"]);self.assertEqual(r["reference_kind"],"resolved_full_body_ROI_prediction")
  self.assertIsNone(r["native_B0_box_IoU"])
  self.assertTrue(g.observe(frame_id=2,native_box=[0,0,10,10],**args)["enabled"])
 def test_resolved_ROI_is_selected_over_conflicting_raw_B0(self):
  g=RegionObservationGate("a")
  r=g.observe(track_id="a",frame_id=1,native_box=[0,0,10,10],b0_box=[50,50,60,60],
   current_B0=True,current_SAM=True,prior_available=True,resolved_box=[0,0,10,10],resolved_current_support=True)
  self.assertTrue(r["enabled"]);self.assertEqual(r["native_reference_box_IoU"],1.)
 def test_history_without_current_observation_cannot_claim_resolved_support(self):
  with self.assertRaises(ValueError):
   RegionObservationGate("a").observe(track_id="a",frame_id=1,native_box=[0,0,10,10],b0_box=None,
    current_B0=False,current_SAM=False,prior_available=True,resolved_box=[0,0,10,10],resolved_current_support=True)
