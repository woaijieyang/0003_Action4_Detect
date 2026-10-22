"""Pure adapter approval/path/command/geometry-gate checks; no Blender scene mocks."""
from pathlib import Path
import sys,unittest,importlib.util,tempfile,json,copy,hashlib
from unittest.mock import patch
R=Path('/data/zy/vln/a-test');O=R/'outputs/action4_modular_bilateral_preview_prepare_v001';sys.dont_write_bytecode=True;sys.path.insert(0,str(R/'src'))
spec=importlib.util.find_spec('data.action4_modular_bilateral_preview');adapter=__import__('data.action4_modular_bilateral_preview',fromlist=['main']) if spec else None
class AdapterTests(unittest.TestCase):
 def ready(self):self.assertIsNotNone(adapter,'new bilateral preview adapter not implemented')
 def test_output_scope_and_fresh_guard(self):
  self.ready()
  scope=adapter.run_root_from_config(adapter.read(R/'configs/data/action4_modular_bilateral_preview_v002.json'))
  for p in [R/'outputs/action4_modular_bilateral_prepare_v001/runs/new',scope,R/'data/new']:
   with self.assertRaises(ValueError):adapter.output_path(p,allowed_runs=scope)
  with tempfile.TemporaryDirectory(dir=O) as temp:
   scope=adapter.run_root_from_config({'output_scope':str((Path(temp)/'runs').relative_to(R))});scope.mkdir();(scope/'old').mkdir()
   self.assertEqual(adapter.output_path(scope/'new',allowed_runs=scope),scope/'new')
   with self.assertRaises(FileExistsError):adapter.output_path(scope/'old',allowed_runs=scope)
  with self.assertRaises(ValueError):adapter.run_root_from_config({'output_scope':'data/not_runtime/runs'})
 def test_fixed_CPU_GPU_command_and_resource_contract(self):
  self.ready();config=R/'configs/data/action4_modular_bilateral_preview_v002.json';out=adapter.run_root_from_config(adapter.read(R/'configs/data/action4_modular_bilateral_preview_v002.json'))/'unit_never_launch';binary=R/'runtime/blender-3.6.23-linux-x64/blender'
  cmd=adapter.build_command(config,out,binary,threads=2,gpu_uuid=None,probe_only=True)
  self.assertIn('--factory-startup',cmd);self.assertIn('--disable-autoexec',cmd);self.assertIn('--background',cmd);self.assertIn('-noaudio',cmd);self.assertIn('--probe-only',cmd);self.assertNotIn('--gpu-uuid',cmd);self.assertEqual(cmd[cmd.index('--threads')+1],'2');self.assertIn('--inside-blender',cmd)
  uuid=sorted(adapter.ALLOWED_GPU_UUIDS)[0];cmd=adapter.build_command(config,out,binary,threads=2,gpu_uuid=uuid,probe_only=False);self.assertEqual(cmd[cmd.index('--gpu-uuid')+1],uuid);self.assertNotIn('--probe-only',cmd)
  for threads,gpu,probe in [(3,None,True),(2,uuid,True),(2,None,False),(2,'GPU-not-allowed',False)]:
   with self.assertRaises(ValueError):adapter.build_command(config,out,binary,threads=threads,gpu_uuid=gpu,probe_only=probe)
 def test_approval_fields_bind_mode_output_and_current_sources(self):
  self.ready();out=adapter.run_root_from_config(adapter.read(R/'configs/data/action4_modular_bilateral_preview_v002.json'))/'unit_never_launch';ref=lambda key:{'path':'outputs/mock/'+key+'.json','sha256':'a'*64}
  bindings={k:ref(k) for k in adapter.APPROVAL_BINDING_KEYS}
  a={'status':adapter.CPU_STATUS,'samples':1,'threads':2,'probe_only':True,'gpu_uuid':None,'output_dir':str(out.relative_to(R)),**bindings,'reviews':{k:{**ref(k),'pass':True} for k in ['spec_review','root_review']}}
  adapter.check_approval_fields(a,bindings,out,threads=2,gpu_uuid=None,probe_only=True)
  for mutate in [lambda b:b.update(samples=2),lambda b:b.update(threads=3),lambda b:b.update(probe_only=False),lambda b:b.update(output_dir='outputs/elsewhere'),lambda b:b['scene_config'].update(sha256='0'*64),lambda b:b['reviews']['spec_review'].update({'pass':False})]:
   bad=copy.deepcopy(a);mutate(bad)
   with self.assertRaises(ValueError):adapter.check_approval_fields(bad,bindings,out,threads=2,gpu_uuid=None,probe_only=True)
 def test_complete_actor_gate_rejects_missing_or_unresolved(self):
  self.ready();keys=['core','skin','band',*adapter.MOUNT_KEYS];cross={a+'__'+b:{'outside_or_unresolved_pairs':[]} for a in keys for b in keys};internal={a+'__'+b:{'outside_or_unresolved_pairs':[]} for i,a in enumerate(sorted(keys)) for b in sorted(keys)[i+1:]}
  report={'cross_pair_count':81,'cross_pairs':cross,'internal_pair_count_each':36,'internal_pairs':{'left':copy.deepcopy(internal),'right':copy.deepcopy(internal)},'outside_or_unresolved_surface_pairs':0};adapter.check_geometry_gate(report)
  for change in ['missing','nonzero','underreported']:
   bad=copy.deepcopy(report)
   if change=='missing':bad['cross_pairs'].pop(next(iter(cross)))
   else:
    bad['internal_pairs']['left'][next(iter(internal))]['outside_or_unresolved_pairs']=[[0,1]]
    if change=='nonzero':bad['outside_or_unresolved_surface_pairs']=1
   with self.assertRaises(ValueError):adapter.check_geometry_gate(bad)
 def test_real_ref_SHA_and_scope_rejection(self):
  self.ready()
  with tempfile.TemporaryDirectory(dir=O) as temp:
   p=Path(temp)/'report.json';p.write_text('{"status":"PASS"}');ref={'path':str(p.relative_to(R)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()};self.assertEqual(adapter.check_ref(ref),p)
   with self.assertRaises(ValueError):adapter.check_ref({**ref,'sha256':'0'*64})
   with self.assertRaises(ValueError):adapter.check_ref({'path':'../outside.json','sha256':'a'*64})
if __name__=='__main__':unittest.main(verbosity=2)
