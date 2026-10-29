import unittest,numpy as np
from scipy.spatial.transform import Rotation
from eval.action4_marker_camera_diagnostic import marker_homographies,fit_hypothesis,orthogonality_residual
class CameraHypothesisTests(unittest.TestCase):
 def test_recovers_exact_synthetic_unequal_rectangles(self):
  rng=np.random.default_rng(43);K=np.array([[1300.,0,1600.],[0,1300,1200],[0,0,1.]]);quads=[]
  for n in range(40):
   X=np.array([[0,0,0],[.03,0,0],[.03,.02,0],[0,.02,0]]);R=Rotation.from_euler('xyz',rng.uniform(-.7,.7,3)).as_matrix();t=np.r_[rng.uniform(-.15,.15,2),rng.uniform(.4,.8)];q=(X@R.T+t)@K.T;quads.append(q[:,:2]/q[:,2:])
  h=marker_homographies(quads);fit=fit_hypothesis(h,np.arange(40),width=3248,height=2464);np.testing.assert_allclose(fit['K'],K,atol=.3,rtol=0)
 def test_invalid_observation_rejected(self):
  with self.assertRaises(ValueError):marker_homographies(np.full((2,4,2),np.nan))
