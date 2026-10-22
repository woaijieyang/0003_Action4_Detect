import sys,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
ROOT=Path('/data/zy/vln/a-test');sys.path.insert(0,str(ROOT/'src'))
from data import action4_left_brace_build as m
class BuildPreparation(unittest.TestCase):
 def test_bound_sources_prepare_and_no_target(self):
  s,c,p,V,F,M,*_=m.prepare();self.assertEqual(c['side'],'left');self.assertEqual(len(V),432);self.assertFalse(m.TARGET.exists());self.assertEqual(s['actual_build_max'],1)
 def test_wrong_output_rejected(self):
  for p in [ROOT/'data/action4_scene_assets/modular_brace_v001/build_v002',ROOT/'outputs/wrong',ROOT.parent/'outside']:
   with self.assertRaises(ValueError):m.target_path(p)
 def test_source_digest_mismatch_rejected(self):
  spec=m.old.load(m.OWN/'build_spec.json');spec['sources']=dict(spec['sources']);spec['sources'][next(iter(spec['sources']))]='0'*64
  with patch.object(m.old,'load',return_value=spec):
   with self.assertRaises(ValueError):m.prepare()
 def test_no_approval_no_BPY(self):
  with self.assertRaises(ValueError):m.check_approval({'status':'not_approved'})
  self.assertNotIn('bpy',sys.modules);self.assertFalse(m.TARGET.exists())
 def test_left_dimensions_not_right(self):
  s,c,p,V,F,M,*_=m.prepare();rig=m.old.load(c['sources']['rig_contract']['path']);self.assertEqual(p['inner'].tolist(),rig['fixed_brace_and_camera_attachment']['left']['brace_inner_radii_m']);self.assertFalse(np.array_equal(p['inner'],rig['fixed_brace_and_camera_attachment']['right']['brace_inner_radii_m']))
if __name__=='__main__':unittest.main(verbosity=2)
