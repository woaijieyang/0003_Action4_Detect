import unittest
import numpy as np
import torch
from data import DataAdapter
from data.action4_dataset import prepare_rois,virtual_roi
from data.action4_roi_contract import validate_expansion,SWEEP_EXPANSIONS
class ROIGeometry(unittest.TestCase):
 @classmethod
 def setUpClass(cls):torch.set_num_threads(1)
 def image(self):
  y,x=np.mgrid[:80,:80];return np.stack([x,y,np.full_like(x,100)],-1).astype(np.uint8)
 def test_six_actual_pixel_samples_and_K(self):
  rgb=self.image();K=np.array([[100.,0,40],[0,120,40],[0,0,1.]])
  for e in SWEEP_EXPANSIONS:
   with self.subTest(e=e):
    out,aa=prepare_rois([rgb],[[20,20,40,30]],[0],roi_size=448,expansion=e)
    pixel=(out[0,:,224,224]*torch.tensor([.229,.224,.225])+torch.tensor([.485,.456,.406]))*255
    np.testing.assert_allclose(pixel.numpy(),[30+10*e/448,25+10*e/448,100],atol=.001)
    A=aa[0];u=np.array([32.,26.,1.]);np.testing.assert_allclose(np.linalg.inv(A)@(A@u),u)
    for size in [112,224,448]:
     down,_=DataAdapter.letterbox_affine((448,448),(size,size));T=down@A
     X=np.array([.1,.2,2.]);native=K@X;native/=native[2]
     projected=(T@K)@X;projected/=projected[2]
     np.testing.assert_allclose(projected,T@native)
 def test_negative_padding_and_no_corner_clipping(self):
  out,aa=prepare_rois([self.image()],[[-8,-8,8,8]],[0],roi_size=448,expansion=1.)
  pix=(out[0,:,0,0]*torch.tensor([.229,.224,.225])+torch.tensor([.485,.456,.406]))*255
  np.testing.assert_allclose(pix.numpy(),0,atol=.0001)
  self.assertEqual(virtual_roi([20,20,40,30],1.),[20.,15.,40.,35.])
  self.assertLess(DataAdapter.transform_points([[-20,-20]],aa[0])[0,0],0)
 def test_instance_order_only_reorders_same_ROIs(self):
  boxes=[[20,20,40,30],[40,40,60,70]];a,A=prepare_rois([self.image()],boxes,[0,0],roi_size=448,expansion=1.1)
  b,B=prepare_rois([self.image()],boxes[::-1],[0,0],roi_size=448,expansion=1.1)
  self.assertTrue(torch.equal(a,b.flip(0)));np.testing.assert_array_equal(A[0],B[1])
 def test_bad_coefficient_rejected(self):
  for v in [True,None,float('nan'),.99]:
   with self.assertRaises(ValueError):validate_expansion(v)
