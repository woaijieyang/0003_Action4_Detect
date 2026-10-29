import unittest
from model.action4_sam_anchor_rotation import retire_previous_anchor
class AnchorTests(unittest.TestCase):
 def setUp(self):
  self.core={"obj_id_to_idx":{0:0},"output_dict":{"cond_frame_outputs":{4:"seed",121:"old",145:"new"},"non_cond_frame_outputs":{}}}
  self.state={"sam2_inference_states":[self.core],"cached_frame_outputs":{4:"seed",121:"old",145:"new"}}
  owner=self
  class NativeStub:
   def clear_all_points_in_frame(self,state,frame,obj,need_output=True):
    owner.assertFalse(need_output);owner.assertEqual(frame,121);owner.assertEqual(obj,0)
    state["output_dict"]["non_cond_frame_outputs"][frame]=state["output_dict"]["cond_frame_outputs"].pop(frame)
  self.tracker=NativeStub()
  self.kw={"obj_id":0,"initial_frame":4,"old_frame":121,"new_frame":145,
   "binding":{"obj_id":0,"frame_index":145,"reason":"bound_unique_prediction_support"}}
 def test_keeps_seed_and_current_removes_old_reference(self):
  r=retire_previous_anchor(self.state,self.tracker,**self.kw)
  self.assertEqual(set(self.core["output_dict"]["cond_frame_outputs"]),{4,145})
  self.assertNotIn(121,self.core["output_dict"]["non_cond_frame_outputs"])
  self.assertNotIn(121,self.state["cached_frame_outputs"])
  self.assertFalse(r["archived_predictions_modified"])
 def test_unverified_binding_cannot_delete(self):
  self.kw["binding"]["reason"]="unbound"
  with self.assertRaises(ValueError):retire_previous_anchor(self.state,self.tracker,**self.kw)
  self.assertIn(121,self.core["output_dict"]["cond_frame_outputs"])
 def test_initial_anchor_cannot_be_deleted(self):
  self.kw["old_frame"]=4
  with self.assertRaises(ValueError):retire_previous_anchor(self.state,self.tracker,**self.kw)
 def test_unknown_extra_anchor_not_silently_deleted(self):
  self.core["output_dict"]["cond_frame_outputs"][80]="unknown"
  with self.assertRaises(ValueError):retire_previous_anchor(self.state,self.tracker,**self.kw)
