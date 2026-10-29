"""CPU surface-to-face contract fixtures, never model accuracy."""
import copy,unittest
from pathlib import Path
import numpy as np
from data.action4_surface_snapshot import make_snapshot
from data.action4_sam_roi_inputs import ImageFrame,array_digest
from data.action4_roi_contract import RoiSpec
from model.action4_face_geometry import FaceGeometryContract
from eval.action4_face_inputs import prepare_face_inputs
R=Path("/data/zy/vln/a-test")
class FaceInputTests(unittest.TestCase):
 def setUp(self):
  self.contract=FaceGeometryContract.from_bytes((R/"configs/model/action4_six_face_anchor_contract_v001.json").read_bytes(),
   (R/"data/DJI_Action4_construct/r3_v001/asset_manifest.json").read_bytes())
  pixels=np.zeros((64,96,3),np.uint8)
  self.frame=ImageFrame("fixture",pixels,"native_integer_centers","a"*64,"native_original",array_digest(pixels))
  self.models={k:{"path":"outputs/action4_six_face_anchor_preparation_v001/CPU_FIXTURE_NOT_MODEL_"+k,"sha256":("b" if k=="A0" else "c")*64} for k in ["A0","B0"]}
  self.roi=RoiSpec.from_full_box([-5,5,55,50],1.5)
  self.pid="b0:fixture:grid:3"
  self.K=np.array([[80.,0,48],[0,80,32],[0,0,1]])
  output={"surface":{"xyz":np.full((1,3,112,112),2.,np.float32),"masks":np.zeros((1,2,112,112),np.float32),
   "regions":np.zeros((1,65,112,112),np.float32)},"rotation":np.eye(3,dtype=np.float32)[None],
   "translation":np.array([[0,0,.5]],np.float32)}
  self.snapshot=make_snapshot(frame=self.frame,prediction={"prediction_id":self.pid,"grid_index":3,"bbox_full_xyxy":list(self.roi.full_body_bbox)},
   output=output,index=0,A112=self.roi.affine(112),A448=self.roi.affine(),K_native=self.K,K112=self.roi.camera(self.K,112).astype(np.float32),
   canonical=np.asarray(self.contract.canonical8_m,np.float32),models=self.models)
 def run_input(self,**kw):
  args=dict(frame=self.frame,expected_models=self.models,expected_prediction_id=self.pid,roi_spec=self.roi,
   contract=self.contract,min_depth_m=1e-6,grazing_cos_threshold=.05);args.update(kw)
  return prepare_face_inputs(self.snapshot,**args)
 def test_actual_XYZ_scale_no_clamp_and_identity(self):
  x=self.run_input();a=self.snapshot.arrays
  np.testing.assert_array_equal(x.xyz_object_m,np.broadcast_to(a["low_m"].astype(float)[:,None,None]+2*a["extent_m"].astype(float)[:,None,None],x.xyz_object_m.shape))
  self.assertTrue((x.xyz_object_m[:,0,0]>np.asarray(self.contract.canonical8_m).max(0)).all())
  self.assertFalse(x.metadata["independent_observation"]);self.assertFalse(x.metadata["semantic_face_labels_available"])
  self.assertEqual(x.geometry_prior["canonical_input_representation"]["dtype"],"float32")
  with self.assertRaises(ValueError):x.xyz_object_m[0,0,0]=0
 def test_wrong_ROI_models_or_instance_rejected(self):
  with self.assertRaises(ValueError):self.run_input(roi_spec=RoiSpec.from_full_box(self.roi.full_body_bbox,2.25))
  m=copy.deepcopy(self.models);m["A0"]["sha256"]="d"*64
  with self.assertRaises(ValueError):self.run_input(expected_models=m)
  with self.assertRaises(ValueError):self.run_input(expected_prediction_id="b0:fixture:grid:9")
 def test_source_pixels_cannot_be_relabelled(self):
  pixels=self.frame.pixels.copy();pixels[0,0,0]=1
  f=ImageFrame("fixture",pixels,"native_integer_centers","a"*64,"native_original",array_digest(pixels))
  with self.assertRaises(ValueError):self.run_input(frame=f)
