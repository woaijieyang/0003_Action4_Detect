import unittest
from model.action4_sam_history_window import prune_history
class WindowTests(unittest.TestCase):
 def state(self):
  value=object()
  return {"sam2_inference_states":[{"output_dict":{"cond_frame_outputs":{0:value},
   "non_cond_frame_outputs":{i:value for i in range(1,100)}},
   "output_dict_per_obj":{0:{"cond_frame_outputs":{0:value},"non_cond_frame_outputs":{i:value for i in range(1,100)}}},
   "temp_output_dict_per_obj":{0:{"non_cond_frame_outputs":{i:value for i in range(1,100)}}},
   "consolidated_frame_inds":{"non_cond_frame_outputs":set(range(1,100))}}],
   "cached_frame_outputs":{i:value for i in range(100)}}
 def test_forward_retains15_and_seed_all_aliases_pruned(self):
  s=self.state();r=prune_history(s,99,after_frame=True)
  core=s["sam2_inference_states"][0];expected=set(range(85,100))
  self.assertEqual(set(core["output_dict"]["non_cond_frame_outputs"]),expected)
  self.assertEqual(set(core["output_dict_per_obj"][0]["non_cond_frame_outputs"]),expected)
  self.assertEqual(set(core["temp_output_dict_per_obj"][0]["non_cond_frame_outputs"]),expected)
  self.assertEqual(core["consolidated_frame_inds"]["non_cond_frame_outputs"],expected)
  self.assertEqual(set(s["cached_frame_outputs"]),expected|{0})
  self.assertFalse(r["archive_modified"])
 def test_reverse_keeps_correct_side_not_far_future(self):
  s=self.state();prune_history(s,20,reverse=True,after_frame=True)
  self.assertEqual(set(s["sam2_inference_states"][0]["output_dict"]["non_cond_frame_outputs"]),set(range(20,35)))
 def test_before_read_keeps15_previous_plus_existing_current(self):
  s=self.state();prune_history(s,99)
  self.assertEqual(set(s["sam2_inference_states"][0]["output_dict"]["non_cond_frame_outputs"]),set(range(84,100)))
 def test_extra_anchors_not_silently_deleted(self):
  s=self.state();s["sam2_inference_states"][0]["output_dict"]["cond_frame_outputs"].update({20:object(),40:object()})
  with self.assertRaises(ValueError):prune_history(s,99)
  self.assertEqual(len(s["sam2_inference_states"][0]["output_dict"]["non_cond_frame_outputs"]),99)
 def test_short_prefix(self):
  s=self.state();prune_history(s,3,after_frame=True)
  self.assertEqual(set(s["sam2_inference_states"][0]["output_dict"]["non_cond_frame_outputs"]),{1,2,3})


class CrossObjectAnchorTests(unittest.TestCase):
 def make_core(self, anchors, ordinary):
  marker=object()
  return {"output_dict":{"cond_frame_outputs":dict.fromkeys(anchors,marker),"non_cond_frame_outputs":dict.fromkeys(ordinary,marker)},
   "output_dict_per_obj":{0:{"cond_frame_outputs":dict.fromkeys(anchors,marker),"non_cond_frame_outputs":dict.fromkeys(ordinary,marker)}},
   "temp_output_dict_per_obj":{0:{"non_cond_frame_outputs":dict.fromkeys(ordinary,marker)}},
   "consolidated_frame_inds":{"non_cond_frame_outputs":set(ordinary)}}
 def assert_domains(self,state,expected,anchors):
  for core in state["sam2_inference_states"]:
   self.assertEqual(set(core["output_dict"]["non_cond_frame_outputs"]),expected)
   self.assertLessEqual(len(core["output_dict"]["non_cond_frame_outputs"]),15)
   for field in ("output_dict_per_obj","temp_output_dict_per_obj"):
    self.assertEqual(set(core[field][0]["non_cond_frame_outputs"]),expected)
   self.assertEqual(core["consolidated_frame_inds"]["non_cond_frame_outputs"],expected)
  self.assertEqual(set(state["cached_frame_outputs"]),expected|set(anchors))
 def test_other_object_correction_is_not_forward_ordinary_anchor(self):
  expected=set(range(21,36))
  corrected=self.make_core([4,20],expected);uncorrected=self.make_core([4],expected|{20})
  native_anchor=corrected["output_dict"]["cond_frame_outputs"][20]
  state={"sam2_inference_states":[corrected,uncorrected],"cached_frame_outputs":dict.fromkeys(expected|{4,20,0})}
  proof=prune_history(state,35,after_frame=True)
  self.assert_domains(state,expected,[4,20])
  self.assertIs(corrected["output_dict"]["cond_frame_outputs"][20],native_anchor)
  self.assertEqual(set(uncorrected["output_dict"]["cond_frame_outputs"]),{4})
  self.assertEqual(proof["anchors"],[4,20])
 def test_other_object_correction_is_not_reverse_ordinary_anchor(self):
  expected=set(range(5,20))
  corrected=self.make_core([4,20],expected);uncorrected=self.make_core([4],expected|{20})
  native_anchor=corrected["output_dict"]["cond_frame_outputs"][20]
  state={"sam2_inference_states":[corrected,uncorrected],"cached_frame_outputs":dict.fromkeys(expected|{4,20,40})}
  prune_history(state,5,reverse=True,after_frame=True)
  self.assert_domains(state,expected,[4,20])
  self.assertIs(corrected["output_dict"]["cond_frame_outputs"][20],native_anchor)
