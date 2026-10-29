import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from model.action4_region_pose_initializers import recent_pose_initializer
class RecentPoseTests(unittest.TestCase):
 def rows(self):
  return [dict(track_id="a",generation=0,pts=p,R=Rotation.from_rotvec([0,0,p*.001]).as_matrix().tolist(),
   t_m=[p*.0001,0,.4],observation_id=str(p),current_image_supported=True,prediction_only=False) for p in [0,10,30,40,60,70]]
 def test_actual_time_last5_constant_motion(self):
  r=recent_pose_initializer(self.rows(),track_id="a",generation=0,target_pts=80,time_base=(1,1000))
  self.assertEqual(r["source_pts"],[10,30,40,60,70]);self.assertEqual(r["history_count"],5)
  np.testing.assert_allclose(r["t_m"],[.008,0,.4],atol=1e-12)
  np.testing.assert_allclose(r["R"],Rotation.from_rotvec([0,0,.08]).as_matrix(),atol=1e-12)
  self.assertFalse(r["published"]);self.assertFalse(r["measurement"])
 def test_prediction_and_other_body_cannot_be_observations(self):
  rows=self.rows();rows[-1]["prediction_only"]=True
  with self.assertRaises(ValueError):recent_pose_initializer(rows,track_id="a",generation=0,target_pts=80,time_base=(1,1000))
  with self.assertRaises(ValueError):recent_pose_initializer(self.rows(),track_id="b",generation=0,target_pts=80,time_base=(1,1000))
 def test_future_and_long_gap_reject(self):
  with self.assertRaises(ValueError):recent_pose_initializer(self.rows(),track_id="a",generation=0,target_pts=70,time_base=(1,1000))
  r=recent_pose_initializer(self.rows(),track_id="a",generation=0,target_pts=1000,time_base=(1,1000));self.assertEqual(r["status"],"unavailable")
class CandidateGateTests(unittest.TestCase):
 def test_backfacing_low_cost_cannot_replace_valid_fresh(self):
  from model.action4_region_pose_initializers import select_clear_seed_region_fit
  c=np.array([[x,y,z] for x in [-.03,.03] for y in [-.015,.015] for z in [-.02,.02]])
  button=np.array([[-.01,0,.02],[.01,0,.02],[0,.01,.02]])
  fresh=dict(R=np.diag([1.,-1.,-1.]).tolist(),t_m=[0,0,.4],optimizer_success=True,at_component_bound=False,final_cost=10.)
  bad=dict(fresh,R=np.eye(3).tolist(),final_cost=1.)
  result=select_clear_seed_region_fit(fresh,bad,c,button)
  self.assertFalse(result["history_selected"]);self.assertIn("observed_top_part_backfacing",result["history_gate"]["reasons"])
 def test_valid_alternative_can_replace_invalid_fresh_even_with_higher_cost(self):
  from model.action4_region_pose_initializers import select_clear_seed_region_fit
  c=np.array([[x,y,z] for x in [-.03,.03] for y in [-.015,.015] for z in [-.02,.02]])
  button=np.array([[-.01,0,.02],[.01,0,.02],[0,.01,.02]])
  fresh=dict(R=np.eye(3).tolist(),t_m=[0,0,.4],optimizer_success=True,at_component_bound=False,final_cost=1.)
  good=dict(fresh,R=np.diag([1.,-1.,-1.]).tolist(),final_cost=10.)
  result=select_clear_seed_region_fit(fresh,good,c,button);self.assertTrue(result["history_selected"])

class FrontPartGateTests(unittest.TestCase):
 def test_front_normal_is_not_silently_treated_as_top_button(self):
  from model.action4_region_pose_initializers import select_clear_seed_region_fit
  c=np.array([[x,y,z] for x in [-.03,.03] for y in [-.015,.015] for z in [-.02,.02]])
  lens=np.array([[-.01,-.016,0],[.01,-.016,0],[0,-.016,.01]])
  good=dict(R=Rotation.from_euler("x",np.pi/2).as_matrix().tolist(),t_m=[0,0,.4],optimizer_success=True,at_component_bound=False,final_cost=5.)
  bad=dict(good,R=Rotation.from_euler("x",-np.pi/2).as_matrix().tolist(),final_cost=1.)
  r=select_clear_seed_region_fit(good,bad,c,lens,observed_normal_object=[0,-1,0])
  self.assertTrue(r["fresh_gate"]["eligible"]);self.assertFalse(r["history_selected"])
  self.assertIn("observed_part_backfacing",r["history_gate"]["reasons"])

class GenerationResetTests(unittest.TestCase):
 def test_new_generation_clears_history_and_stale_generation_is_retained_as_error(self):
  import tempfile,os
  from pathlib import Path
  from unittest.mock import patch
  from data.action4_paths import PROJECT_ROOT
  from eval.action4_part_region_sequence import fit_track
  root=PROJECT_ROOT/".cache";root.mkdir(exist_ok=True)
  with tempfile.TemporaryDirectory(dir=root) as temp:
   p=Path(temp)/"input.npz";c=np.array([[x,y,z] for x in [-.03,.03] for y in [-.015,.015] for z in [-.02,.02]])
   R=np.diag([1.,-1.,-1.]);t=[0.,0.,.4]
   np.savez(p,body=np.ones((8,8),bool),button=np.ones((8,8),bool),vertices=c,button_vertices=c,
    K=np.eye(3),marker_xyz=c[:4],marker_uv=np.zeros((4,2)))
   jobs=[]
   for i,g in enumerate([0,1,1,0]):
    jobs.append(dict(frame_id=i,pts=i*400,time_base=[1,11988],track_id="a",body_generation=g,
      status="prepared",use_recent_history=True,crop_native=[0,0,8,8],input={"path":str(p),"sha256":"fixture"},
      A0_raw={"corners_raw_from_resolved_roi":[[0,0]]*8},K_native=np.eye(3).tolist(),canonical8=c.tolist(),
      initial_pose={"R":R.tolist(),"t_m":t},part_receipt={"mask_identity":"fixture"+str(i)},
      parameters={"rotation_limit_rad":.8,"translation_limit_cad":.6}))
   fit=dict(R=R.tolist(),t_m=t,optimizer_success=True,at_component_bound=False,final_cost=1.)
   with patch("eval.action4_part_region_sequence.ReceiptVerifier.verify",return_value=p),patch("eval.action4_part_region_sequence.refine_clear_seed_regions",return_value=fit):
    out=fit_track((jobs,min(os.sched_getaffinity(0))))
   self.assertEqual([r["status"] for r in out],["candidate","candidate","candidate","fit_error"])
   self.assertTrue(out[0]["history_generation_reset"]);self.assertTrue(out[1]["history_generation_reset"])
   self.assertEqual(out[1]["history_initialization"]["prior"]["reason"],"no_current_image_history")
   self.assertEqual(out[2]["history_initialization"]["prior"]["history_count"],1)
   self.assertIn("Stale body generation",out[3]["reason"])

if __name__=="__main__":unittest.main()
