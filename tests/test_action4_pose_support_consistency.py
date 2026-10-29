import unittest
import numpy as np
from model.action4_pose_support_consistency import pose_support_consistency,SupportPolicy
K=np.array([[800.,0,128],[0,800,128],[0,0,1.]])
CAD=np.array([[x,y,z] for x in [-.035,.035] for y in [-.022,.022] for z in [-.016,.016]])
class SupportConsistencyTests(unittest.TestCase):
 def mask(self):
  import cv2
  xyz=CAD+np.array([0,0,.5]);h=xyz@K.T;uv=h[:,:2]/h[:,2,None]
  a=np.zeros((256,256),np.uint8);cv2.fillConvexPoly(a,cv2.convexHull(np.rint(uv).astype(np.int32)),1)
  return a.astype(bool)
 def call(self,mask,r=None,**kw):
  return pose_support_consistency(mask,K,CAD,np.eye(3) if r is None else r,[0,0,.5],policy=SupportPolicy(),**kw)
 def test_correct_outline_compatible_not_pose_gt(self):
  d=self.call(self.mask());self.assertTrue(d['reference_compatible']);self.assertFalse(d['canonical_ID_accuracy_measured'])
 def test_wrong_long_axis_not_trusted(self):
  from model.action4_native_rotation_history import rotation_exp
  d=self.call(self.mask(),rotation_exp([0,0,np.pi/2]))
  self.assertFalse(d['reference_compatible']);self.assertIn('visible_support_outside_projected_body',d['reasons'])
 def test_partial_visibility_unknown_not_false_target(self):
  m=self.mask();m[:140]=False
  d=self.call(m);self.assertFalse(d['reference_compatible']);self.assertIn('insufficient_visible_shape_evidence',d['reasons'])
  self.assertGreater(d['visible_coverage'],.99)
 def test_180_symmetry_does_not_prove_ID(self):
  from model.action4_native_rotation_history import rotation_exp
  a=self.call(self.mask());b=self.call(self.mask(),rotation_exp([0,0,np.pi]))
  self.assertTrue(a['reference_compatible']);self.assertTrue(b['reference_compatible'])
  self.assertFalse(b['canonical_ID_accuracy_measured'])
 def test_other_instance_not_owned_support(self):
  d=self.call(self.mask(),other_instance_mask=self.mask())
  self.assertFalse(d['available']);self.assertFalse(d['reference_compatible'])
 def test_empty_not_perfect_coverage(self):
  d=self.call(np.zeros((256,256),bool));self.assertIsNone(d['visible_coverage']);self.assertFalse(d['reference_compatible'])
 def test_bad_rotation_and_depth_fail(self):
  with self.assertRaises(ValueError):self.call(self.mask(),2*np.eye(3))
  with self.assertRaises(ValueError):pose_support_consistency(self.mask(),K,CAD,np.eye(3),[0,0,-.5],policy=SupportPolicy())

class RigidSupportTests(unittest.TestCase):
 def test_unknown_display_region_cannot_be_declared_rigid(self):
  from model.action4_video_motion_observation import measure_motion
  a=np.zeros((100,100),np.uint8);m=np.ones((100,100),bool)
  r=measure_motion(a,a,m,m,[0,0,100,100],require_dynamic_exclusion=True)
  self.assertFalse(r['available']);self.assertEqual(r['reason'],'dynamic_display_exclusion_unknown')
 def test_all_texture_on_dynamic_display_is_excluded(self):
  from model.action4_video_motion_observation import measure_motion
  import cv2
  rng=np.random.default_rng(31);a=np.zeros((160,200),np.uint8)
  a[40:120,50:150]=(rng.random((80,100))*255).astype(np.uint8)
  b=cv2.warpAffine(a,np.float32([[1,0,3],[0,1,2]]),(200,160))
  m=np.ones_like(a,bool);dynamic=np.zeros_like(a,bool);dynamic[35:130,45:160]=1
  r=measure_motion(a,b,m,m,[0,0,200,160],previous_dynamic=dynamic,current_dynamic=dynamic,require_dynamic_exclusion=True)
  self.assertFalse(r['available'])
 def test_shape_input_bound_to_current_frame(self):
  from model.action4_video_pose_stabilizer import VideoPoseStabilizer,StabilizerConfig
  from test_action4_video_pose_stabilizer import packet,K,CAD
  s=VideoPoseStabilizer(StabilizerConfig(motion_veto_enabled=True,require_reference_shape=True))
  with self.assertRaises(ValueError):
   s.apply_prediction(packet(0),frame_id=0,pts=0,time_base=[1,30],K_native=K,canonical_m=CAD,mode='video',
      motion_observations={},current_supports=dict(frame_id=1,pts=1,time_base=[1,30],masks={}))
 def test_unqualified_texture_does_not_veto(self):
  from model.action4_video_pose_stabilizer import VideoPoseStabilizer,StabilizerConfig
  from test_action4_video_pose_stabilizer import packet,K,CAD
  a=VideoPoseStabilizer(StabilizerConfig());b=VideoPoseStabilizer(StabilizerConfig(motion_veto_enabled=True,require_rigid_texture=True))
  for f in range(10):
   kw=dict(frame_id=f,pts=f,time_base=[1,30],K_native=K,canonical_m=CAD,mode='video')
   x=a.apply_prediction(packet(f,angle=.2*f),**kw)
   observation=dict(track_id='a',frame_id=f,pts=f,time_base=[1,30],available=True,rigid_support_qualified=False)
   y=b.apply_prediction(packet(f,angle=.2*f),motion_observations={'a':observation},**kw)
   np.testing.assert_array_equal(x['instances'][0]['corners_final'],y['instances'][0]['corners_final'])

class ReferenceConsumptionTests(unittest.TestCase):
 def test_same_RGB_pose_and_motion_different_owned_mask_changes_veto(self):
  import cv2
  from model.action4_video_pose_stabilizer import VideoPoseStabilizer,StabilizerConfig
  from test_action4_video_pose_stabilizer import packet,K,CAD
  from model.action4_native_rotation_history import rotation_exp
  def mask(rotation):
   xyz=CAD@rotation.T+np.array([0,0,1]);h=xyz@K.T;uv=h[:,:2]/h[:,2,None]
   m=np.zeros((800,1000),np.uint8);cv2.fillConvexPoly(m,cv2.convexHull(np.rint(uv).astype(np.int32)),1)
   return m.astype(bool)
  masks=[mask(np.eye(3)),mask(rotation_exp([0,0,np.pi/2]))]
  engines=[VideoPoseStabilizer(StabilizerConfig(motion_veto_enabled=True,require_reference_shape=True,require_rigid_texture=True)) for _ in range(2)]
  uv=np.array([[x,y] for y in [390.,396.,402.,408.] for x in [480.,490.,500.,510.,520.]])
  decisions=[[],[]];outputs=[[],[]]
  for f in range(12):
   obs=dict(track_id='a',frame_id=f,previous_frame_id=f-1,pts=f,time_base=[1,30],same_generation=True,
      available=True,rigid_support_qualified=True,points=len(uv),inlier_ratio=1.,coverage=.6,previous_uv=uv.tolist(),current_uv=uv.tolist())
   for i,(engine,m) in enumerate(zip(engines,masks)):
    out=engine.apply_prediction(packet(f,angle=1.7 if f>=6 else 0),frame_id=f,pts=f,time_base=[1,30],
       K_native=K,canonical_m=CAD,mode='video',motion_observations={'a':obs} if f else {},
       current_supports=dict(frame_id=f,pts=f,time_base=[1,30],masks={'a':m}))['instances'][0]
    decisions[i].append(out['pose_stabilization']['image_motion_decision'])
    outputs[i].append(out['final_pose']['rotation'])
  self.assertTrue(any(d.get('reject_candidate') for d in decisions[0]))
  self.assertTrue(any(d.get('reason')=='historical_reference_not_shape_compatible' for d in decisions[1]))
  self.assertFalse(np.array_equal(outputs[0],outputs[1]))
 def test_missing_current_mask_cannot_support_hold(self):
  from model.action4_video_pose_stabilizer import VideoPoseStabilizer,StabilizerConfig
  from test_action4_video_pose_stabilizer import packet,K,CAD
  s=VideoPoseStabilizer(StabilizerConfig(motion_veto_enabled=True,require_reference_shape=True,motion_veto_hold_frames=5))
  out=s.apply_prediction(packet(0),frame_id=0,pts=0,time_base=[1,30],K_native=K,canonical_m=CAD,mode='video',
      motion_observations={},current_supports=dict(frame_id=0,pts=0,time_base=[1,30],masks={}))
  d=out['instances'][0]['pose_stabilization']['image_motion_decision']
  self.assertFalse(d['reference_consistency']['reference_compatible'])
  self.assertFalse(d.get('bounded_rotation_hold',False))

class HullExclusionRegression(unittest.TestCase):
 def test_hole_between_dynamic_regions_cannot_supply_rigid_points(self):
  from model.action4_video_motion_observation import measure_motion
  rng=np.random.default_rng(59)
  image=np.zeros((160,200),np.uint8);image[45:115,60:140]=(rng.random((70,80))*255).astype(np.uint8)
  dynamic=np.zeros_like(image,bool);dynamic[25:135,30:40]=True;dynamic[25:135,160:170]=True
  body=np.ones_like(image,bool)
  out=measure_motion(image,image,body,body,[0,0,200,160],previous_dynamic=dynamic,current_dynamic=dynamic,require_dynamic_exclusion=True)
  self.assertFalse(out["available"])
