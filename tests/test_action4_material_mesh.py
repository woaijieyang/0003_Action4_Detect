import unittest,numpy as np
import json,hashlib
from pathlib import Path
from model.action4_material_mesh import closest_triangle_points,MaterialMeshBinder
class MaterialMeshTests(unittest.TestCase):
 def mesh(self,component=0,reverse=False):
  vertices=np.array([[0,0,0],[0,1,0],[1,0,0]],float);face=np.array([[0,1,2] if not reverse else [0,2,1]])
  return MaterialMeshBinder(vertices,face,np.array([component]),np.array([[.2,.3,0]]),mesh_identity='synthetic_triangle_m',excluded_components=[6])
 def args(self):return (np.array([.2,.3,.001]),1,np.eye(3),np.array([0,0,1]),np.diag([100.,100,1]),np.array([20.,30.]))
 def test_double_precision_closest_point_and_barycentric(self):
  tri=np.array([[[0,0,0],[1,0,0],[0,1,0]]],float)
  for point,expected in [([.2,.3,1],[.2,.3,0]),([1,1,0],[.5,.5,0]),([-1,-1,0],[0,0,0])]:
   q,bary,d=closest_triangle_points(point,tri);np.testing.assert_allclose(q[0],expected,atol=1e-14);np.testing.assert_allclose(bary.sum(1),1);self.assertTrue((bary>=0).all());np.testing.assert_allclose(np.einsum('ni,nij->nj',bary,tri),q,atol=1e-14)
 def test_valid_binding_is_only_provisional(self):
  b=self.mesh().bind(*self.args(),max_reprojection_px=2)
  self.assertEqual(b['status'],'provisional');self.assertFalse(b['geometry_accepted']);np.testing.assert_allclose(b['X_cad_m'],[.2,.3,0])
 def test_glass_backface_distance_and_pose_inconsistency_reject(self):
  self.assertEqual(self.mesh(6).bind(*self.args(),max_reprojection_px=2)['reason'],'display_or_glass_semantic_exclusion')
  self.assertEqual(self.mesh(reverse=True).bind(*self.args(),max_reprojection_px=2)['reason'],'back_facing_surface')
  args=list(self.args());args[0]=np.array([.2,.3,1.]);self.assertEqual(self.mesh().bind(*args,max_reprojection_px=2)['reason'],'outside_surface_distance')
  args=list(self.args());args[-1]=np.array([50,30]);self.assertEqual(self.mesh().bind(*args,max_reprojection_px=2)['reason'],'pose_surface_projection_inconsistent')
 def test_corner_two_faces_not_silently_projected(self):
  v=np.array([[0,0,0],[0,1,0],[1,0,0],[0,0,1]],float);f=np.array([[0,1,2],[0,3,1]])
  m=MaterialMeshBinder(v,f,np.array([0,0]),np.array([[0,.3,0]]),mesh_identity='synthetic_two_faces')
  a=list(self.args());a[0]=np.array([0,.3,0]);a[-1]=np.array([0,30])
  self.assertEqual(m.bind(*a,max_reprojection_px=2)['reason'],'multiple_face_normal_ambiguity')
 def test_invalid_geometry_rejected_before_projection(self):
  a=list(self.args());a[2]=np.zeros((3,3))
  with self.assertRaises(ValueError):self.mesh().bind(*a,max_reprojection_px=2)
  a=list(self.args());a[0][0]=float('nan')
  with self.assertRaises(ValueError):self.mesh().bind(*a,max_reprojection_px=2)
 def test_competing_face_just_outside_distance_gate_still_causes_ambiguity(self):
  v=np.array([[0,0,0],[0,1,0],[1,0,0],[0,0,.0565],[1,0,.0565],[0,1,.0565]],float)
  m=MaterialMeshBinder(v,np.array([[0,1,2],[3,4,5]]),np.array([0,0]),np.array([[.2,.3,0]]),mesh_identity='thin_shell_fixture')
  point=np.array([.2,.3,.0275]);self.assertLess(.0275,m.distance);self.assertGreater(.029,m.distance);self.assertLess(.029-.0275,m.ambiguity)
  a=list(self.args());a[0]=point
  self.assertEqual(m.bind(*a,max_reprojection_px=2)['reason'],'multiple_face_normal_ambiguity')
 def test_formal_policy64regions_and_complete_identity(self):
  root=Path('/data/zy/vln/a-test');path=root/'configs/model/action4_material_mesh_policy_v001.json';m=MaterialMeshBinder.from_policy(str(path))
  self.assertEqual(m.regions.shape,(64,3));self.assertIn('policy_sha256',m.binding_provenance);self.assertIn('regions_sha256',m.binding_provenance);self.assertIn('semantic_source_sha256',m.binding_provenance)
  out=root/'outputs/action4_material_mesh_preparation_v001';cfg=json.loads(path.read_text());cfg['ambiguity_normal_angle_deg']=10
  changed=out/'changed_policy_test.json';changed.write_text(json.dumps(cfg));other=MaterialMeshBinder.from_policy(str(changed));self.assertNotEqual(m.binding_identity,other.binding_identity)
  rc=out/'regions63_test.json';rc.write_text(json.dumps({'xyz_object_m':m.regions[:63].tolist()}));cfg['surface_regions']=str(rc);cfg['surface_regions_sha256']=hashlib.sha256(rc.read_bytes()).hexdigest();changed.write_text(json.dumps(cfg))
  with self.assertRaisesRegex(ValueError,'64'):MaterialMeshBinder.from_policy(str(changed))

if __name__=='__main__':unittest.main(verbosity=2)
