import ast,copy,unittest
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from data.action4_roi_contract import RoiSpec
from data import action4_dataset as dataset
from test_action4_sam_supported_roi import fixture,context,observed,DummyPipeline

class SharedSpec(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(1)
  path=Path("/data/zy/vln/a-test/runtime/action4_runs/p2_a0_roi15_prediction50_v001/source/src/data/action4_dataset.py")
  tree=ast.parse(path.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="prepare_rois")
  namespace=dict(dataset.__dict__);exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),"exec"),namespace)
  cls.old=staticmethod(namespace["prepare_rois"])
 def test_actual_sampler_bitwise_parent_parity(self):
  rng=np.random.default_rng(419);rgb=torch.from_numpy(rng.integers(0,256,(3,64,96),dtype=np.uint8))
  for n in [112,224,448]:
   # Existing sampler sizes are multiples of32;112 is the downstream geometry grid.
   if n%32:continue
   for e in [1.,1.5,2.25]:
    boxes=[[-12.5,-8.5,30.5,40.5],[30,10,80,55]]
    old,A=self.old([rgb],boxes,[0,0],roi_size=n,expansion=e)
    new,B,specs=dataset.prepare_rois([rgb],boxes,[0,0],roi_size=n,expansion=e,return_specs=True)
    self.assertTrue(torch.equal(old,new))
    for a,b,s in zip(A,B,specs):
     np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(s.affine(),b)
 def test_tampering_and_no_double_expansion(self):
  s=RoiSpec.from_full_box([10,20,50,40],1.5)
  self.assertEqual(s.continuous_bounds,(0.,0.,60.,60.))
  with self.assertRaises(ValueError):replace(s,continuous_bounds=(0,0,90,90))
  for n in [112,224,448]:
   K=np.array([[300,0,123],[0,310,70],[0,0,1.]])
   np.testing.assert_array_equal(s.camera(K,n),s.affine(n)@K)
 def test_same_body_box_same_A0_tensor_with_or_without_track(self):
  f=fixture();a=DummyPipeline();b=DummyPipeline()
  a.pipeline.roi_expansion=b.pipeline.roi_expansion=1.5
  box=f[0]["fused"]["tracks"][0]["bbox_for_roi"]
  a.pipeline._detect=lambda rgb:[{"score":.8,"bbox_full_xyxy":box}]
  x=a.run(f[1],f[2]);y=b.run(f[1],f[2],context(f))
  self.assertTrue(torch.equal(a.rois[-1],b.rois[-1]));self.assertTrue(torch.equal(a.K[-1],b.K[-1]))
  self.assertEqual(x["a0_roi_specs"][0]["continuous_bounds"],y["a0_roi_specs"][0]["continuous_bounds"])
  self.assertEqual(y["a0_roi_specs"][0]["full_body_bbox"],box)
  self.assertIsNone(y["instances"][0]["raw_bbox_full_xyxy"])
 def test_mask_bbox_is_not_substituted(self):
  f=fixture();p=DummyPipeline();p.pipeline.roi_expansion=1.5
  r=p.run(f[1],f[2],context(f))
  ys,xs=np.nonzero(f[4]);visible=[float(xs.min()),float(ys.min()),float(xs.max()+1),float(ys.max()+1)]
  self.assertNotEqual(r["a0_roi_specs"][0]["full_body_bbox"],visible)
