import json,unittest,hashlib
from pathlib import Path
import numpy as np
from eval.action4_native_m3t_adapter import native_float32_se3_for_projection,SE3_REPRESENTATION_POLICY
from model.action4_face_geometry import FaceGeometryContract,project_six_faces
from data.action4_paths import PROJECT_ROOT as ROOT
class NativeSE3RepresentationTests(unittest.TestCase):
 def setUp(self):
  out=ROOT/"outputs/action4_m3t_native_track_808_822_v001";self.rows=[json.loads(x) for x in (out/"POSES.jsonl").read_text().splitlines()]
  cbytes=(ROOT/"configs/model/action4_six_face_anchor_contract_v001.json").read_bytes();c=json.loads(cbytes);self.contract=FaceGeometryContract.from_bytes(cbytes,(ROOT/c["parent_asset"]["path"]).read_bytes());self.c=np.array(c["canonical8_m"]);self.K=np.array(json.loads((out/"INPUT_MANIFEST.json").read_text())["rows"][0]["K"])
 def check(self,T):return native_float32_se3_for_projection(T,self.c,self.K)
 def test_actual_nine_drift_cases_preserve_raw_and_pass_original_geometry(self):
  drift=[r for r in self.rows if r.get("numeric_error")];self.assertEqual(len(drift),9)
  for row in drift:
   T=np.load(ROOT/row["pose_native_file"]);before=T.copy();result=self.check(T);np.testing.assert_array_equal(T,before);np.testing.assert_array_equal(result["raw_matrix"].astype(float),T);U=result["converted_matrix"];np.testing.assert_array_equal(U[:3,3],T[:3,3]);self.assertLessEqual(result["max_corner_projection_change_px"],.01)
   project_six_faces(self.c,self.contract,U[:3,:3],U[:3,3],self.K,units="m",pixel_space_id="raw_video_native_integer_centers",image_size_wh=[3248,2464],min_depth_m=1e-6,grazing_cos_threshold=.05)
 def test_gross_nonrigid_reflection_nonfinite_bottom_rejected(self):
  T=np.eye(4,dtype=np.float32);T[2,3]=.5
  bad=[]
  for scale in [1.0001,1.01,-1.]:
   b=T.copy();b[0,0]=scale;bad.append(b)
  b=T.copy();b[0,1]=.001;bad.append(b)
  b=T.copy();b[0,0]=np.nan;bad.append(b)
  b=T.copy();b[3,0]=1e-8;bad.append(b)
  for b in bad:
   with self.subTest(matrix=b.tolist()),self.assertRaises(ValueError):self.check(b)
 def test_arbitrary_double_pose_not_silently_quantized(self):
  T=np.eye(4);T[2,3]=.50000000001
  with self.assertRaises(ValueError):self.check(T)
 def test_positive_depth_and_finite_K_gates(self):
  T=np.eye(4,dtype=np.float32);T[2,3]=-.5
  with self.assertRaises(ValueError):self.check(T)
  T[2,3]=.5;K=self.K.copy();K[0,0]=np.inf
  with self.assertRaises(ValueError):native_float32_se3_for_projection(T,self.c,K)
 def test_registered_policy_exact_and_original_module_unchanged(self):
  self.assertEqual(json.loads((ROOT/"configs/model/action4_native_float32_se3_representation_v001.json").read_text()),SE3_REPRESENTATION_POLICY)
  manifest=json.loads((ROOT/"outputs/action4_front_region_lens_assembly_v001/SOURCE_MANIFEST.json").read_text());ref=manifest["src/model/action4_face_geometry.py"]
  self.assertEqual(hashlib.sha256((ROOT/ref["path"]).read_bytes()).hexdigest(),ref["sha256"])
