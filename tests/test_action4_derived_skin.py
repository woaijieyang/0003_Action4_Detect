import copy,json,unittest
from data import action4_global_scene as m
class DerivedSkinConfigTests(unittest.TestCase):
 def setUp(self):
  self.raw=json.loads((m.ROOT/'configs/data/action4_global_scene_init_v001.json').read_text())
  self.derived=json.loads((m.ROOT/'configs/data/action4_global_scene_periodic_skin_v001.json').read_text())
 def test_v1_raw_unchanged(self):
  m.validate_config(self.raw);self.assertEqual(m.skin_input_spec(self.raw)['color_space'],'sRGB')
 def test_v2_explicit_reviewed_linear(self):
  m.validate_config(self.derived);s=m.skin_input_spec(self.derived)
  self.assertEqual(s['color_space'],'Linear');self.assertEqual(s['source_photo'],self.raw['sources']['skin_forearm'])
 def test_v2_without_binding_stays_raw(self):
  c=copy.deepcopy(self.raw);c['schema']='action4.global_scene_init.v2';m.validate_config(c)
  self.assertEqual(m.skin_input_spec(c)['mode'],'original_photo')
 def test_v1_cannot_silently_gain_derived_field(self):
  c=copy.deepcopy(self.derived);c['schema']='action4.global_scene_init.v1'
  with self.assertRaises(ValueError):m.validate_config(c)
 def test_wrong_colour_space_and_malformed_binding_reject(self):
  for value in (None,{}, {'image':self.derived['derived_skin']['image'],'manifest':self.derived['derived_skin']['manifest'],'color_space':'sRGB'}):
   c=copy.deepcopy(self.derived);c['derived_skin']=value
   with self.assertRaises(ValueError):m.validate_config(c)
 def test_image_and_manifest_hash_mismatch_reject(self):
  for key in ('image','manifest'):
   c=copy.deepcopy(self.derived);c['derived_skin'][key]['sha256']='0'*64
   with self.assertRaises(ValueError):m.validate_config(c)
if __name__=='__main__':unittest.main(verbosity=2)
