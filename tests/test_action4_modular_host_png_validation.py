"""Runtime boundary tests with real host Pillow; no Blender/project scene/GPU."""
from pathlib import Path
import copy,hashlib,json,sys,unittest
from unittest.mock import patch
from PIL import Image
R=Path('/data/zy/vln/a-test');sys.dont_write_bytecode=True;sys.path.insert(0,str(R/'src'))
from data import action4_modular_bilateral_preview as adapter
from data import action4_modular_pixels as pixels
from data.action4_adapter import DataAdapter
from test_action4_modular_pixels import harness

class HostPNGTests(unittest.TestCase):
 def test_child_structure_only_and_pending_receipt(self):
  original=DataAdapter.validate_sample;flags=[]
  def structure_only(a,record,*,check_image=True):
   flags.append(check_image)
   if check_image:raise RuntimeError('Blender has no PIL')
   return original(a,record,check_image=False)
  with harness(self,[1,2],output_role='diagnostic') as h,patch.object(DataAdapter,'validate_sample',structure_only):
   result=pixels.render_frame(**h.args)
   self.assertEqual(flags,[False]);self.assertTrue(result['pending_host_png_validation']);self.assertEqual(result['render_calls'],4)
   self.assertTrue((h.run/'samples/mock_frame/render_receipt.json').is_file())
   self.assertTrue((h.run/'samples/mock_frame/record.json').is_file())
 def test_host_real_PNGs_SHA_and_dimensions(self):
  with harness(self,[1,2],output_role='diagnostic') as h:
   result=pixels.render_frame(**h.args);check=pixels.validate_host_pngs(result['record'])
   self.assertEqual(len(check['files']),5);self.assertTrue(check['PNG_decoded_and_dimensions_verified']);self.assertTrue(check['file_SHA256_verified'])
 def test_host_rejects_corrupt_PNG_even_when_hash_matches(self):
  with harness(self,[2],output_role='diagnostic') as h:
   record=pixels.render_frame(**h.args)['record'];path=h.root/record['image'];path.write_bytes(b'not a PNG');record['rgb_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
   with self.assertRaises(ValueError):pixels.validate_host_pngs(record)
 def test_host_rejects_RGB_or_mask_dimensions_even_when_hash_matches(self):
  for role in ['rgb','visible','amodal']:
   with self.subTest(role=role),harness(self,[2],output_role='diagnostic') as h:
    record=pixels.render_frame(**h.args)['record'];row=record['instances'][0]
    key,hashkey=('image','rgb_sha256') if role=='rgb' else (('mask_visible','mask_visible_sha256') if role=='visible' else ('mask_amodal_expanded','mask_amodal_expanded_sha256'))
    owner=record if role=='rgb' else row;path=h.root/owner[key];Image.new('L',(7,5)).save(path);owner[hashkey]=hashlib.sha256(path.read_bytes()).hexdigest()
    with self.assertRaises(ValueError):pixels.validate_host_pngs(record)
 def test_host_rejects_changed_file_SHA(self):
  with harness(self,[2],output_role='diagnostic') as h:
   record=pixels.render_frame(**h.args)['record'];Image.new('RGB',(32,24),'red').save(h.root/record['image'])
   with self.assertRaisesRegex(ValueError,'SHA'):pixels.validate_host_pngs(record)
 def test_host_publication_requires_validation_and_retains_failure(self):
  for broken in [False,True]:
   with self.subTest(broken=broken),harness(self,[1,2],output_role='diagnostic') as h,patch.object(adapter,'ROOT',h.root):
    pixel=pixels.render_frame(**h.args);out=h.data.parent;bindings={'mock':'CPU_MOCK_ONLY'}
    child={'status':'BILATERAL_RENDER_COMPLETE_PENDING_HOST_PNG_VALIDATION','pending_host_png_validation':True,'probe_only':False,'rendered':True,'render_calls':4,'pixel_result':pixel,'bindings':bindings,'source_identities':h.args['source_identities'],'final_scene_and_source_verified':True,'dataset_generation_allowed':False,'training_accepted':False}
    adapter.write(out/'render_receipt.json',child)
    if broken:(h.root/pixel['record']['image']).write_bytes(b'corrupt after child')
    if broken:
     with self.assertRaises(ValueError):adapter.finalize_host_render(out,bindings)
     self.assertFalse((out/'diagnostic_report.json').exists());self.assertTrue((out/'failure.json').is_file());self.assertTrue((out/'render_receipt.json').is_file())
    else:
     result=adapter.finalize_host_render(out,bindings)
     self.assertFalse(result['pending_host_png_validation']);self.assertTrue((out/'diagnostic_report.json').is_file());self.assertFalse(result['training_accepted'])
if __name__=='__main__':unittest.main(verbosity=2)
