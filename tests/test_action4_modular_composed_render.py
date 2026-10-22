"""CPU/cache validation only. No bpy, scene creation, pose, build or GPU."""
from pathlib import Path
import sys,unittest,copy,ast
from unittest.mock import patch
sys.dont_write_bytecode=True
ROOT=Path('/data/zy/vln/a-test');sys.path.insert(0,str(ROOT/'src'))
from data import action4_modular_composed_render as composed
class ComposedPreparationTests(unittest.TestCase):
 def test_real_inputs_contact_and_exact_v003(self):
  c,ctx=composed.prepare();self.assertEqual(ctx['registry']['verified_files'],18)
  self.assertLess(ctx['geometry']['capture_error_m'],1e-12)
  self.assertEqual(c['environment'],ctx['neutral_config']['environment'])
  self.assertEqual(c['render'],ctx['neutral_config']['render'])
  self.assertEqual(ctx['geometry']['mod']['uv_0'].shape,(51840,2))
  self.assertEqual(ctx['geometry']['y']['source_loop_normals'].shape,(51840,3))
  receipt=composed.read(c['sources']['mount_y25_contact']['path']);self.assertEqual(receipt['surface_pairs'],0);self.assertEqual(receipt['checked_pairs'],6)
  self.assertNotIn('bpy',sys.modules)
 def test_wrong_frame_side_threads_and_photometry_rejected(self):
  original=composed.read;c=original(composed.CONFIG)
  for key,value in [('frame_id',540),('instance_id',1),('threads',3),('environment',{**c['environment'],'world_strength':9})]:
   bad=copy.deepcopy(c);bad[key]=value
   with patch.object(composed,'read',side_effect=lambda p:bad if composed.bounded(p)==composed.CONFIG else original(p)):
    with self.assertRaises(ValueError):composed.prepare()
 def test_frozen_digest_and_bad_contact_receipt_rejected(self):
  realsha=composed.sha
  with patch.object(composed,'sha',side_effect=lambda p:'bad' if str(p).endswith('y_plus25.npz') else realsha(p)):
   with self.assertRaises(ValueError):composed.prepare()
  original=composed.read;c=original(composed.CONFIG);contactpath=c['sources']['mount_y25_contact']['path'];bad=copy.deepcopy(original(contactpath));bad['status']='BLOCKED_NEW_MOUNT_Y25_CONTACT_REQUIRES_REVIEW';bad['unresolved_surface_pairs']=1
  with patch.object(composed,'read',side_effect=lambda p:bad if str(p)==contactpath else original(p)):
   with self.assertRaises(ValueError):composed.prepare()
 def test_new_output_boundary_and_no_automatic_reuse(self):
  for p in [ROOT/'outputs/action4_modular_render_prepare_v001/runs/a',composed.OWN/'runs',ROOT/'src']:
   with self.assertRaises(ValueError):composed.output_path(p,False)
  self.assertEqual(composed.output_path(composed.OWN/'runs/future',False),composed.OWN/'runs/future')
 def test_verified_pixel_math_AST_reused_with_ID_parameter(self):
  old=ast.parse((ROOT/'src/data/action4_modular_render.py').read_text());new=ast.parse(Path(composed.__file__).read_text())
  of=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=='execute_blender');nf=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='_render_target_pixels')
  class OriginalID(ast.NodeTransformer):
   def visit_Name(self,n):return ast.copy_location(ast.Constant(value=2),n) if n.id=='instance_id' else n
  APIs=['render_helpers.project_canonical','canonical.build_export','canonical.verify_export','render_helpers.crop_geometry_preflight','amodal.render_crop','render_helpers.mask_measurements','synth._render_device','synth._set_denoiser','base._mask_nodes']
  for api in APIs:
   left=[n for n in ast.walk(of) if isinstance(n,ast.Call) and ast.unparse(n.func)==api]
   right=[n for n in ast.walk(nf) if isinstance(n,ast.Call) and ast.unparse(n.func)==api]
   self.assertEqual(len(left),1,api);self.assertEqual(len(right),1,api)
   self.assertEqual(ast.dump(left[0],include_attributes=False),ast.dump(OriginalID().visit(copy.deepcopy(right[0])),include_attributes=False),api)
  self.assertIn('instance_id',[a.arg for a in nf.args.args]);self.assertNotIn('exec(',Path(composed.__file__).read_text())
if __name__=='__main__':unittest.main(verbosity=2)
