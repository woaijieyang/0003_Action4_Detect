import unittest,numpy as np
from scipy.spatial.transform import Rotation
from model.action4_marker_orientation_candidate import candidates
class MarkerOrientationTests(unittest.TestCase):
 def test_exact_projected_square_and_full_body_recovers_pose(self):
  K=np.array([[1300.,0,1600],[0,1300,1200],[0,0,1]]);C=np.array([[x,y,z] for x in [-.035,.035] for y in [-.016,.016] for z in [-.022,.022]]);R=Rotation.from_euler('xyz',[1.7,.2,2.9]).as_matrix();t=np.array([.06,.03,.5]);M=np.array([[-1.,0,0],[0,0,-1],[0,-1,0]]);U=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]])*.02;X=U@M.T+[.02,.016,.015]
  proj=lambda a:((a@R.T+t)@K.T)[:,:2]/((a@R.T+t)@K.T)[:,2:]
  uv=proj(C);box=np.r_[uv.min(0),uv.max(0)];out=candidates(proj(X),K,C,box,marker_aspect=1.)
  self.assertTrue(out);best=min(out,key=lambda a:a['marker_reprojection_rms_px']);np.testing.assert_allclose(best['rotation'],R,atol=1e-5);np.testing.assert_allclose(best['translation_m'],t,atol=1e-5)
  self.assertFalse(best['geometry_accepted']);self.assertEqual(best['corner_ids'],list(range(8)))
