import tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from data.action4_paths import PROJECT_ROOT as R
from eval.action4_image_guided_eval import _is_run_launcher_alias
class LauncherAliasTests(unittest.TestCase):
 def test_only_bound_readonly_bootstrap_alias(self):
  base=R/'outputs/action4_stage4_full650_engineering_v001';base.mkdir(exist_ok=True)
  with tempfile.TemporaryDirectory(dir=base) as d:
   d=Path(d);root=d/'source';root.mkdir();launch=d/'launch.py';launch.write_text('# fixture');launch.chmod(0o444);spec=d/'RUN_SPEC.json';spec.write_text('{}')
   m=SimpleNamespace(__name__='__main__',__spec__=None,__file__=str(launch),spec_path=spec,code=root)
   self.assertTrue(_is_run_launcher_alias('__mp_main__',m,root))
   self.assertFalse(_is_run_launcher_alias('model.foreign',m,root))
   m.code=d;self.assertFalse(_is_run_launcher_alias('__mp_main__',m,root));m.code=root
   m.__file__=str(root/'foreign.py');self.assertFalse(_is_run_launcher_alias('__mp_main__',m,root));m.__file__=str(launch)
   launch.chmod(0o644);self.assertFalse(_is_run_launcher_alias('__mp_main__',m,root))

 def test_torch_virtual_namespaces_are_not_project_files(self):
  import torch
  from eval.action4_image_guided_eval import _is_project_origin
  for name,module in [('torch.ops',torch.ops),('torch.classes',torch.classes)]:
   origin=Path(module.__file__).resolve()
   self.assertFalse(origin.is_file())
   self.assertFalse(_is_project_origin(name,origin))
 def test_real_foreign_project_file_still_classified_for_rejection(self):
  from eval.action4_image_guided_eval import _is_project_origin
  self.assertTrue(_is_project_origin('foreign_actual_file',R/'src/eval/action4_image_guided_eval.py'))
  self.assertTrue(_is_project_origin('model.foreign',R/'nonexistent.py'))

 def test_framework_proxy_requires_exact_template_and_owned_location(self):
  from unittest.mock import patch
  import torch.distributed.nn.jit.instantiator as gen
  from eval.action4_image_guided_eval import _is_verified_torch_proxy
  base=R/'outputs/action4_stage4_full650_engineering_v003'
  with tempfile.TemporaryDirectory(dir=base) as d:
   d=Path(d);name='_remote_module_non_scriptable';path=d/(name+'.py')
   fields=dict(assign_module_interface_cls='module_interface_cls = None',args='*args',kwargs='**kwargs',arg_types='*args, **kwargs',arrow_and_return_type='',arrow_and_future_return_type='',jit_script_decorator='')
   text=gen.get_remote_module_template(True).format(**fields);path.write_text(text)
   mod=SimpleNamespace(__name__=name,__file__=str(path))
   with patch.object(gen,'INSTANTIATED_TEMPLATE_DIR_PATH',str(d)):
    self.assertTrue(_is_verified_torch_proxy(name,mod))
    self.assertFalse(_is_verified_torch_proxy('_remote_module_unknown',mod))
    path.write_text(text+'# changed')
    self.assertFalse(_is_verified_torch_proxy(name,mod))
    path.write_text(text);mod.__file__=str(d/'different.py')
    self.assertFalse(_is_verified_torch_proxy(name,mod))

 def test_all_foreign_real_modules_are_reported_and_rejected(self):
  from unittest.mock import patch
  from eval import action4_image_guided_eval as mod
  base=R/'outputs/action4_stage4_full650_engineering_v003'
  with tempfile.TemporaryDirectory(dir=base) as d:
   d=Path(d);root=d/'source';root.mkdir();own=root/'guard.py';own.write_text('# fixture')
   modules={}
   for name in ['foreign_one','foreign_two']:
    p=d/(name+'.py');p.write_text('# fixture');modules[name]=SimpleNamespace(__file__=str(p))
   with patch.object(mod,'sys',SimpleNamespace(modules=modules)),patch.object(mod,'__file__',str(own)):
    with self.assertRaisesRegex(ValueError,'foreign_one.*foreign_two'):mod._origins(root)
