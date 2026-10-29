import unittest
from model.action4_sam_recovery_gate import RecoveryGate
class RecoveryTests(unittest.TestCase):
 def obs(self,f,score=.95):
  return {"track_id":"right","status":"observed","candidate_id":f"b0:frame{f}:grid:7",
   "raw_detection":{"score":score,"refined_quality_score":.9,"grid_index":7,"bbox_full_xyxy":[10,10,30,30]}}
 def test_requires_five_actual_observations_and_native_commit(self):
  g=RecoveryGate("right")
  for f in range(5):r=g.observe(f,"missing",self.obs(f))
  self.assertEqual(r["action"],"propose_native_rebinding");self.assertFalse(r["identity_accepted"])
  self.assertEqual(g.generation,0);g.resolve(r["event_id"],binding_accepted=True);self.assertEqual(g.generation,1)
 def test_no_per_frame_reprompt_when_pending(self):
  g=RecoveryGate("right")
  for f in range(5):r=g.observe(f,"missing",self.obs(f))
  self.assertEqual(g.observe(5,"missing",self.obs(5))["action"],"await_native_binding")
 def test_inference_failure_is_not_target_loss(self):
  g=RecoveryGate("right")
  for f in range(10):r=g.observe(f,"inference_error",self.obs(f))
  self.assertEqual(r["action"],"no_automatic_correction");self.assertIsNone(g.pending)
 def test_weak_candidate_does_not_complete_chain(self):
  g=RecoveryGate("right")
  for f in range(8):r=g.observe(f,"missing",self.obs(f,score=.8))
  self.assertEqual(r["action"],"collect_evidence")
 def test_frame_gap_breaks_confirmation(self):
  g=RecoveryGate("right")
  for f in range(4):g.observe(f,"missing",self.obs(f))
  r=g.observe(6,"missing",self.obs(6));self.assertEqual(r["confirmed_observations"],1)
 def test_rejected_binding_does_not_advance_generation(self):
  g=RecoveryGate("right")
  for f in range(5):r=g.observe(f,"missing",self.obs(f))
  g.resolve(r["event_id"],binding_accepted=False);self.assertEqual(g.generation,0)
  for f in range(5,10):r=g.observe(f,"missing",self.obs(f))
  self.assertNotEqual(r["action"],"propose_native_rebinding")
