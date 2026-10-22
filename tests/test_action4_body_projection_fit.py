"""Body-only math, topology and support checks; not real-data fit acceptance."""
import unittest,numpy as np
from data.action4_body_projection_fit import *
class BodyProjectionTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.m=prepare()
 def test_only_reviewed_frame0_bilateral(self):
  self.assertEqual({c["id"] for c in self.m["cases"]},{"f0_L","f0_R"})
  self.assertEqual(RATIOS,(.6,.8125,1.1))
 def test_real_component4_edges_and_injective_graph(self):
  edges={tuple(sorted((int(a),int(b)))) for f in self.m["faces"][self.m["components"]==4] for a,b in zip(f,np.roll(f,-1))}
  for feature in self.m["features"].values():
   allpairs=[tuple(sorted(p)) for ps in feature["actual_mesh_vertex_edge_pairs"].values() for p in ps]
   self.assertTrue(all(p in edges for p in allpairs));self.assertEqual(len(allpairs),len(set(allpairs)))
   self.assertEqual(set(feature["actual_mesh_vertex_edge_pairs"]),set(ORDER))
 def test_right_rear_view_means_negative_model_x(self):
  v=self.m["vertices"]
  for feature in self.m["features"].values():
   right=v[np.array(feature["actual_mesh_vertex_edge_pairs"]["right"])].mean((0,1))
   left=v[np.array(feature["actual_mesh_vertex_edge_pairs"]["left"])].mean((0,1))
   self.assertLess(right[0],left[0])
 def test_mesh_physical_scale_unchanged(self):
  np.testing.assert_allclose(np.ptp(self.m["vertices"],axis=0),[.07050054168701172,.03242378234863281,.044552218437194824],atol=1e-10)
 def test_known_pose_homogeneous_projection(self):
  q=np.array([.2,-.1,.3,.1,.05,.5]);T=pose(q);points=self.m["vertices"][::353];f=2639.
  K=np.array([[f,0,1623.5],[0,f,1231.5],[0,0,1]])
  hp=np.column_stack([points,np.ones(len(points))])@(K@T[:3]).T
  np.testing.assert_allclose(project(points,T,f),hp[:,:2]/hp[:,2,None],atol=1e-10)
 def test_proper_pose(self):
  T=pose([.2,-.3,1.,.1,.2,.6]);self.assertAlmostEqual(np.linalg.det(T[:3,:3]),1)
  np.testing.assert_allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-12)
 def test_signed_polygon_distance(self):
  polygon=np.array([[0,0],[1,0],[1,1],[0,1]],float)
  d=polygon_signed_distance(np.array([[.5,.5],[2,.5],[1,.3]]),polygon)
  np.testing.assert_allclose(d,[-.5,1,0],atol=1e-12)
 def test_unknown_boundary_outward_extrusion(self):
  case=self.m["cases"][0];poly,center,sectors=uncertainty(case)
  pts=np.array(case["uncertain_boundary_segments"][0]["points_px"])
  distant=center+(pts-center)*5
  self.assertTrue(unknown_angle(distant,center,sectors).all())
 def test_actual_triangle_support_counts_visible_false_negatives(self):
  case=self.m["cases"][0];feature=self.m["features"]["outer_support"];f=3248*.8125;q=initial(case,f,feature)
  q[3]+=.06
  r=raster(self.m,pose(q),f,case)
  self.assertIsNotNone(r);self.assertGreater(r["metrics"]["certain_visible_false_negative_fraction"],.05)
  self.assertGreater(r["metrics"]["known_background_excess_relative_visible_area"],.05)
 def test_no_invented_fixed_observed_vertices(self):
  for case in self.m["cases"]:
   for edge in case["rear_outer_face_ordered_edge_graph"]["segments"]:
    self.assertIsNone(edge["fixed_model_vertices"]);self.assertFalse(edge["segment_endpoints_are_corners"])
if __name__=="__main__":unittest.main(verbosity=2)
