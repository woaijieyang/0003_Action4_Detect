import sys,unittest,importlib.util,numpy as np
from pathlib import Path
ROOT=Path('/data/zy/vln/a-test');sys.path.insert(0,str(ROOT/'src'))
from data import action4_left_brace_module as m, action4_canonical_export as canonical
class Test(unittest.TestCase):
 def test_source_pointwise(self):
  c=m.load_config();p,V,F,M,*_=m.expected(c);r,T=m.verify_geometry(c,V,F,M);self.assertLess(max(r['pointwise_ring_errors_m'].values()),1e-12);self.assertEqual(T.shape,(864,3))
 def test_wrong_right_radius_rejected(self):
  c=m.load_config();p,V,F,M,*_=m.expected(c);rig=m.base.load(c['sources']['rig_contract']['path']);r=rig['fixed_brace_and_camera_attachment']['right']['brace_inner_radii_m'];wrong=m.base.annular_geometry(r,p['thickness'],.034,72,p['trim_width'])[0]
  with self.assertRaises(ValueError):m.verify_geometry(c,wrong,F,M)
 def test_scale_wrong_face_rejected(self):
  c=m.load_config();p,V,F,M,*_=m.expected(c)
  with self.assertRaises(ValueError):m.verify_geometry(c,V*1.01,F,M)
  bad=F.copy();bad[0]=bad[0,::-1]
  with self.assertRaises(ValueError):m.verify_geometry(c,V,bad,M)
 def test_build_disabled(self):
  with self.assertRaises(RuntimeError):m.build()
 def test_source_scope_sharedK_and_reflection(self):
  sp=importlib.util.spec_from_file_location('prep',ROOT/'outputs/action4_bilateral_f0_prepare_v001/prepare.py');p=importlib.util.module_from_spec(sp);sp.loader.exec_module(p);c=p.load();self.assertEqual(len(c['left']['branches']),2);self.assertIsNone(c['select_branch']);self.assertEqual(c['frame_id'],0)
  with self.assertRaises(ValueError):canonical._rigid(np.diag([-1,1,1,1]),'mirror')
if __name__=='__main__':unittest.main(verbosity=2)
