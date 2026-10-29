import copy,importlib,importlib.util,unittest,numpy as np
def track(center=(150.,150.),size=(100.,100.)):
 return {'track_id':'camera','status':'sam_supported','bbox_for_roi':[90.,90.,210.,210.],'candidate_id':None,'raw_detection':None,'geometry_trace':{'unsmoothed_measurement_center':list(center),'trusted_size_median':list(size),'velocity_native_px_per_second':[0.,0.]}}
class BodyProjectionTests(unittest.TestCase):
 def new(self):
  self.assertIsNotNone(importlib.util.find_spec('model.action4_body_box_projection'),'separate body-size state missing')
  return importlib.import_module('model.action4_body_box_projection').BodyBoxProjection()
 def test_remote_mask_extent_cannot_pull_box_to_frame_bottom(self):
  p=self.new();p.update(0,0.,track(),[110.,110.,190.,190.])
  out=p.update(1,1/30,track(),[110.,110.,400.,600.])
  np.testing.assert_allclose(out['bbox_for_roi'],[100,100,200,200])
  self.assertEqual(out['body_box_projection']['enclosure_decision'],'rejected_extent_conflict')
 def test_accepted_enclosure_does_not_feed_back_into_size_state(self):
  p=self.new();out=p.update(0,0.,track(),[95.,100.,205.,200.])
  self.assertEqual(out['bbox_for_roi'],[95.,100.,205.,200.])
  out=p.update(1,1/30,track(),[110.,110.,190.,190.])
  np.testing.assert_allclose(out['bbox_for_roi'],[100,100,200,200],atol=1e-10)
 def test_gradual_real_size_growth_can_cover_current_support(self):
  p=self.new()
  for f in range(20):
   size=100*1.01**f;t=track(size=(size,size));b=[150-size/2,150-size/2,150+size/2,150+size/2]
   out=p.update(f,f/30,t,b)
   self.assertLessEqual(out['bbox_for_roi'][0],b[0]+1e-9);self.assertGreaterEqual(out['bbox_for_roi'][2],b[2]-1e-9)
 def test_lost_is_not_fabricated_as_current_support(self):
  p=self.new();t=track();t.update(status='lost',bbox_for_roi=None)
  r=p.update(0,0.,t,None);self.assertIsNone(r['bbox_for_roi']);self.assertFalse(r['body_box_projection']['applied'])
 def test_input_raw_detection_and_history_are_not_modified(self):
  p=self.new();t=track();before=copy.deepcopy(t);r=p.update(0,0.,t,[110,110,190,190])
  self.assertEqual(t,before);self.assertEqual(r['raw_detection'],t['raw_detection'])
 def test_nonmonotonic_time_and_nonfinite_measurements_fail(self):
  p=self.new();p.update(0,0.,track(),None)
  with self.assertRaises(ValueError):p.update(1,0.,track(),None)
  p=self.new();t=track(center=(float('nan'),150))
  with self.assertRaises(ValueError):p.update(0,0.,t,None)
if __name__=='__main__':unittest.main()
