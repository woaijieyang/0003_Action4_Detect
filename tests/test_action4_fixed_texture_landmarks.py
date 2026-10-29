import json,copy,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from model.action4_region_pose_initializers import FixedTextureLandmarks,refine_clear_seed_regions,texture_group_residual
from eval.action4_roi_sweep import binding
from data.action4_paths import PROJECT_ROOT as ROOT
class FixedTextureTests(unittest.TestCase):
 def setUp(self):
  self.ref=binding(ROOT/"outputs/action4_face_texture_probe_v001/FIXED_RAY_BINDINGS.json")
  doc=json.loads((ROOT/self.ref["path"]).read_text());self.row=doc["rows"][0];self.seed=self.row["seed"]
  self.points=[p for p in self.row["points"] if p["status"]=="provisional_fixed_binding"]
  self.job=json.loads((ROOT/"outputs/action4_front_region_partial_v002/JOBS.json").read_text())[0]
 def args(self,n=6):
  return dict(track_id="track_00",generation=self.seed["generation"],frame_id=808,keypoint_ids=[p["keypoint_id"] for p in self.points[:n]],pixels_native=[p["native_uv"] for p in self.points[:n]],crop_origin=self.job["crop_native"][:2])
 def context(self):
  return dict(track_id="track_00",generation=self.seed["generation"],frame_id=808,crop_origin=self.job["crop_native"][:2],pixel_space_id="raw_video_native_integer_centers",binding_sha256=self.ref["sha256"])
 def test_arbitrary_N_dimensions(self):
  with np.load(ROOT/self.job["input"]["path"]) as z:
   parts=[dict(part_id="body",vertices_m=z["vertices"],mask=z["body"],partial_boundary=True,observed_boundary_eligibility=z["body_edge_eligibility"])]
   for n in [6,13]:
    group=FixedTextureLandmarks.from_binding(self.ref,**self.args(n))
    r=refine_clear_seed_regions(z["K"],self.seed["R"],self.seed["t_m"],parts,cad_diagonal_m=.09,max_nfev=1,landmark_groups=[group],texture_context=self.context())
    self.assertEqual(r["residual_count"],96+2*n);self.assertEqual(r["landmark_residual_counts"],[2*n])
 def test_identity_duplicates_pixels_and_nonfinite_rejected(self):
  for key,value in [("track_id","track_01"),("generation",999),("pixel_space_id","resized"),("keypoint_ids",[self.points[0]["keypoint_id"]]*6),("pixels_native",[[float("nan"),0]]*6)]:
   args=self.args();args[key]=value
   with self.subTest(key=key),self.assertRaises(ValueError):FixedTextureLandmarks.from_binding(self.ref,**args)
  bad=dict(self.ref,sha256="0"*64)
  with self.assertRaises(ValueError):FixedTextureLandmarks.from_binding(bad,**self.args())
 def test_context_cannot_transfer(self):
  group=FixedTextureLandmarks.from_binding(self.ref,**self.args());c=self.context();c["frame_id"]=809
  with self.assertRaises(ValueError):group.validate_context(c)
 def test_count_doubling_preserves_robust_group_mass(self):
  errors=np.arange(12,dtype=float).reshape(6,2)*3
  def cost(e):
   r=texture_group_residual(e,3.,96,1.);return np.sum(2*(np.sqrt(1+r*r)-1))
  self.assertAlmostEqual(cost(errors),cost(np.repeat(errors,2,axis=0)),places=10)


class SmallResidualPrecisionTests(__import__("unittest").TestCase):
    def test_tiny_correspondence_residual_keeps_local_derivative(self):
        import numpy as np
        from model.action4_region_pose_initializers import texture_group_residual
        e=np.full((6,2),1e-10)
        actual=texture_group_residual(e,3.,96,1.)
        expected=(e/3.*np.sqrt(96/e.size)).ravel()
        np.testing.assert_allclose(actual,expected,rtol=1e-14,atol=0)
        np.testing.assert_array_equal(texture_group_residual(np.zeros((6,2)),3.,96,1.),np.zeros(12))
