import unittest,copy,numpy as np
from model.action4_direct_corner_tracking import direct_native_observations,DirectCornerTracker

def example():
 b=[-10.,20.,50.,60.];l,t,r,bot=-25.,-5.,65.,85.;A=np.array([[224/90,0,-224*l/90-.5],[0,224/90,-224*t/90-.5],[0,0,1.]])
 inst=dict(prediction_id='p',track_id='t',frame_id=5,pts=2000,roi_decision_id='r',bbox_for_a0=b,raw_network_pose={'bad':'must not read'},corners_final=[[999,999]]*8)
 obs=dict(identity=['image','p','roi','spec'],corner_ids=list(range(8)),uv224=np.arange(16).reshape(8,2).tolist(),p_valid=[.1]*8,q_corner=[.1]*8,b224=[1.]*8,multimodal=[False]*8)
 rec=dict(prediction_id='p',track_id='t',frame_id=5,pts=2000,roi_decision_id='r',bbox_for_a0=b,roi_identity=obs['identity'],observations=obs,A224_from_native=A.tolist())
 spec=dict(prediction_id='p',track_id='t',full_body_bbox=b,continuous_bounds=[l,t,r,bot],A224=A.tolist(),expansion=1.5,output_size=448,pixel_convention='native_integer_centres_continuous_crop_edges')
 return inst,rec,spec
class DirectCornerTests(unittest.TestCase):
 def test_half_pixel_padding_inverse_and_no_pose_dependency(self):
  i,r,s=example();out=direct_native_observations(i,r,s);expected=np.asarray(r['observations']['uv224']);expected=(expected+.5)*90/224+[-25,-5]
  np.testing.assert_allclose(out['corners_detector_native'],expected);self.assertLess(expected[0,0],0);self.assertFalse(out['coordinates_clamped'])
  i['raw_network_pose']=None;i['corners_final']=None;i['decoder_diagnostics']={'raise':'never read'}
  self.assertEqual(direct_native_observations(i,r,s),out)
 def test_instance_ID_and_affine_mismatch_fail(self):
  i,r,s=example();bad=copy.deepcopy(s);bad['track_id']='other'
  with self.assertRaises(ValueError):direct_native_observations(i,r,bad)
  bad=copy.deepcopy(r);bad['observations']['corner_ids']=[1,0,2,3,4,5,6,7]
  with self.assertRaises(ValueError):direct_native_observations(i,bad,s)
  bad=copy.deepcopy(s);bad['A224'][0][2]+=.5
  with self.assertRaises(ValueError):direct_native_observations(i,r,bad)
 def update(self,tracker,f,points,box=(0,0,100,100),tid='t'):
  return tracker.update(track_id=tid,frame_id=f,time_seconds=f/30,points_native=points,full_box=box,corner_ids=list(range(8)))
 def test_ROI_motion_and_scale_transported_without_extra_lag(self):
  t=DirectCornerTracker();points=np.arange(16).reshape(8,2)+20
  for f in range(10):
   scale=1+f*.05;shift=np.array([f*10,f*5]);box=np.r_[shift,shift+100*scale];out=self.update(t,f,points*scale+shift,box)
   np.testing.assert_allclose(out['corners_tracked_native'],points*scale+shift,atol=1e-10)
 def test_point_impulse_suppressed_IDs_not_reassigned(self):
  t=DirectCornerTracker();points=np.arange(16).reshape(8,2)+20
  for f in range(5):self.update(t,f,points)
  bad=points.copy();bad[2]+=[50,40];out=self.update(t,5,bad)
  np.testing.assert_allclose(out['corners_tracked_native'],points);self.assertEqual(out['corner_ids'],list(range(8)))
 def test_instances_and_missing_frame_reset(self):
  t=DirectCornerTracker();a=np.ones((8,2))*10;b=np.ones((8,2))*90
  for f in range(6):self.update(t,f,a,tid='a');self.update(t,f,b,tid='b')
  out=self.update(t,8,b,tid='a');self.assertEqual(out['status'],'initialized');np.testing.assert_allclose(out['corners_tracked_native'],b)
 def test_sustained_point_motion_is_followed(self):
  t=DirectCornerTracker();out=None
  for f in range(60):out=self.update(t,f,np.ones((8,2))*(20+f*.3))
  self.assertLess(abs(np.array(out['corners_tracked_native'])[0,0]-(20+59*.3)),2.)
