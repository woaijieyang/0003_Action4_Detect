"""Toy math/identity tests, deliberately separate from actual real-video fitting."""
import unittest,math
import numpy as np
from scipy.spatial.transform import Rotation
from data.action4_rig_projection_fit import (transform,apply,project,segment_deltas,obb_gap,prepare,observations,
 init_pose,unpack,residual_rows,physical,prior,sparsity,TRACKS)
class RigProjectionFitTests(unittest.TestCase):
 def test_proper_se3_no_scale(self):
  T=transform([.3,-.2,.4,.1,.2,.7]);self.assertAlmostEqual(np.linalg.det(T[:3,:3]),1)
  np.testing.assert_allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-12)
 def test_known_forward_projection_only_not_real_pnp(self):
  p=np.array([[0,0,1],[.1,-.2,2]])
  uv=project(p,2000,[3248,2464])
  np.testing.assert_allclose(uv,[[1623.5,1231.5],[1723.5,1031.5]])
 def test_nearest_segment_keeps_latent_sliding_not_vertex_identity(self):
  delta=segment_deltas(np.array([[.2,0],[.8,0]]),np.array([[[0,0],[1,0]],[[0,1],[1,1]]]))
  np.testing.assert_allclose(delta[:,0],0,atol=1e-12)
 def test_sat_rejects_intersection_and_containment(self):
  a=np.eye(4);b=np.eye(4)
  self.assertLess(obb_gap(a,np.ones(3),b,np.ones(3)*.1),0)
  b[0,3]=3;self.assertGreater(obb_gap(a,np.ones(3),b,np.ones(3)),0)
 def test_fixed_attachment_composition_invariant(self):
  A=transform([1.57,0,0,0,0,.062])
  for rot in [[0,0,0],[.4,-.2,.3]]:
   B=transform([*rot,.1,.2,.6]);C=B@A
   np.testing.assert_allclose(np.linalg.inv(B)@C,A,atol=1e-12)
 def test_body_scale_changes_projection_not_an_allowed_fit_variable(self):
  a=np.array([[.035,0,1.]])
  self.assertNotEqual(project(a,2000,[3248,2464])[0,0],project(a*np.array([2,1,1]),2000,[3248,2464])[0,0])
 def test_actual_input_identity_and_no_hidden_correspondence(self):
  m=prepare()
  self.assertFalse(m["contract"]["dataset_generation_allowed"])
  self.assertEqual(m["obs"]["fit_frames"],[0,540,960,1290])
  for f in m["obs"]["frames"]:
   for o in f["observations"]:self.assertIsNone(o.get("model_points_3d_mm"))
 def test_heldout_selector_excludes_body_lens_mount(self):
  m=prepare();rr=observations(m,m["obs"]["held_out_frames"],True)
  self.assertTrue(rr);self.assertTrue(all(o["kind"] in {"brace_visible_edge","forearm_axis"} for _,o in rr))
  self.assertFalse(set(m["obs"]["held_out_frames"])&set(m["obs"]["fit_frames"]))
 def test_61_parameter_sparse_residual_contract(self):
  m=prepare();frames=m["obs"]["fit_frames"];rows=observations(m,frames);f=3248*.8;q=[math.log(f)]
  for t in TRACKS:q.extend(np.r_[Rotation.from_matrix(m["sides"][t]["A0"][:3,:3]).as_rotvec(),m["sides"][t]["A0"][:3,3]])
  for fr in frames:
   for t in TRACKS:q.extend(init_pose(observations(m,[fr]),t,f,m["wh"]))
  q=np.array(q);ff,aa,bb=unpack(q,frames);d,_=residual_rows(m,rows,ff,aa,bb);ph,_=physical(m,aa,bb);pr=prior(q,m)
  self.assertEqual(len(q),61);self.assertEqual(sparsity(m,rows,frames).shape,(len(d)+len(ph)+len(pr),61))
  self.assertTrue(np.isfinite(np.r_[d,ph,pr]).all())
if __name__=="__main__":unittest.main(verbosity=2)
