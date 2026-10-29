import importlib,importlib.util,unittest,numpy as np
def raw(f,ids=(0,1)):
 m=np.zeros((len(ids),12,16),bool)
 for i in range(len(ids)):m[i,3:8,1+i*8:6+i*8]=True
 return {'frame_index':f,'outputs':{'out_obj_ids':np.array(ids,dtype=np.int64),'out_binary_masks':m,'out_probs':np.ones(len(ids)),'out_boxes_xywh':np.zeros((len(ids),4))}}
class ReplayBindingTests(unittest.TestCase):
 def new(self):
  self.assertIsNotNone(importlib.util.find_spec('model.action4_sam_generation'),'replay validator missing')
  C=importlib.import_module('model.action4_sam_generation').CandidateReplay
  return C(session_id='new',generation=1,seed_frame=3,trigger_frame=5,native_obj_map={'left':0,'right':1},height=12,width=16)
 def test_contiguous_context_and_exact_binding(self):
  c=self.new()
  for f in [3,4,5]:c.consume('new',1,raw(f))
  self.assertEqual(c.binding(),{'session_id':'new','generation':1,'frame_id':5,'native_obj_map':{'left':0,'right':1},'status':'native_binding_validated'})
 def test_wrong_session_generation_frame_and_duplicate_are_rejected(self):
  for sid,g,f in [('old',1,3),('new',0,3),('new',1,4)]:
   c=self.new()
   with self.assertRaises(ValueError):c.consume(sid,g,raw(f))
  c=self.new();c.consume('new',1,raw(3))
  with self.assertRaises(ValueError):c.consume('new',1,raw(3))
 def test_missing_context_instance_allowed_but_current_missing_not_committable(self):
  c=self.new();c.consume('new',1,raw(3,(0,)));c.consume('new',1,raw(4));c.consume('new',1,raw(5,(0,)))
  with self.assertRaises(ValueError):c.binding()
 def test_shared_mask_is_not_two_instances(self):
  c=self.new()
  for f in [3,4]:c.consume('new',1,raw(f))
  r=raw(5);r['outputs']['out_binary_masks'][1]=r['outputs']['out_binary_masks'][0];c.consume('new',1,r)
  with self.assertRaises(ValueError):c.binding()
 def test_empty_current_unknown_id_and_nonfinite_are_errors(self):
  c=self.new()
  with self.assertRaises(ValueError):c.consume('new',1,raw(3,(0,3)))
  c=self.new();r=raw(3);r['outputs']['out_probs'][0]=np.nan
  with self.assertRaises(ValueError):c.consume('new',1,r)
 def test_no_early_commit_or_late_frame(self):
  c=self.new();c.consume('new',1,raw(3))
  with self.assertRaises(ValueError):c.binding()
  for f in [4,5]:c.consume('new',1,raw(f))
  with self.assertRaises(ValueError):c.consume('new',1,raw(6))
if __name__=='__main__':unittest.main()
