import unittest,importlib.util
from pathlib import Path
import numpy as np
from model.action4_sam_supported_boxes import geometry_support
class GeometrySpeed(unittest.TestCase):
 def test_exact_original_components_pixels_and_metadata(self):
  path=Path('/data/zy/vln/a-test/outputs/action4_sam_geometry_speed_v001/BEFORE.py')
  spec=importlib.util.spec_from_file_location('geometry_before_speed',path);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
  rng=np.random.default_rng(123);masks=[np.zeros((30,40),bool),np.ones((30,40),bool)]
  for density in [.01,.1,.5]:
   for i in range(10):masks.append(rng.random((61,73))<density)
  mask=np.zeros((100,160),bool);mask[40:80,70:130]=1;mask[0,159]=1;mask[90,1]=1;masks.append(mask)
  for m in masks:
   for fraction in [0,.005,.05]:
    a,ra=old.geometry_support(m,minimum_component_fraction=fraction);b,rb=geometry_support(m,minimum_component_fraction=fraction)
    self.assertEqual(ra,rb);np.testing.assert_array_equal(a,b)
if __name__=='__main__':unittest.main()
