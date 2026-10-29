import unittest,importlib,importlib.util,copy
def event():
 return {'trigger_frame':7,'seed_frame':4,'source_generation':0,'proposed_generation':1,'source_session_id':'s0','new_session':'s1','outcome':'native_bound','native_obj_map':{'track_00':0,'track_01':1},'binding':{'session_id':'s1','generation':1,'frame_id':7,'native_obj_map':{'track_00':0,'track_01':1},'status':'native_binding_validated'},'previous_close':{'historical_closed_session':'s0'}}
class Ownership(unittest.TestCase):
 def functions(self):
  self.assertIsNotNone(importlib.util.find_spec('eval.action4_reanchor_roi_provider'),'source-bound ROI provider missing')
  m=importlib.import_module('eval.action4_reanchor_roi_provider');return m.generation_owners,m.check_frame_owner
 def test_generation_boundary_uses_new_session(self):
  build,check=self.functions();owners=build('s0',[event()])
  self.assertEqual(check({'frame':{'global_frame_id':6},'generation':0,'session_id':'s0'},owners)['session_id'],'s0')
  self.assertEqual(check({'frame':{'global_frame_id':7},'generation':1,'session_id':'s1'},owners)['session_id'],'s1')
 def test_old_session_at_new_generation_is_rejected(self):
  build,check=self.functions()
  with self.assertRaises(ValueError):check({'frame':{'global_frame_id':7},'generation':1,'session_id':'s0'},build('s0',[event()]))
 def test_duplicate_or_foreign_generation_rejected(self):
  build,_=self.functions()
  with self.assertRaises(ValueError):build('s0',[event(),event()])
  e=event();e['source_generation']=2
  with self.assertRaises(ValueError):build('s0',[e])
 def test_swapped_ID_binding_or_missing_close_rejected(self):
  build,_=self.functions()
  for key,value in [('native_obj_map',{'track_00':1,'track_01':0}),('previous_close',{'is_success':False}),('outcome','failed')]:
   e=event();e[key]=value
   with self.assertRaises(ValueError):build('s0',[e])
if __name__=='__main__':unittest.main()
