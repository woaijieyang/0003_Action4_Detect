"""Mock the observed Blender libraries.load in-place list mutation."""
import sys,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from data import action4_modular_bilateral_scene as m
class FakeObject:
 def __init__(self,name):self.name=name
class LoadContext:
 def __init__(self,names,data):self.available=SimpleNamespace(objects=list(names));self.loaded=SimpleNamespace(objects=None);self.data=data;self.assigned_list=None
 def __enter__(self):return self.available,self.loaded
 def __exit__(self,*args):
  self.assigned_list=self.loaded.objects;objects=[FakeObject(n) for n in self.assigned_list];self.assigned_list[:]=objects
  for o in objects:self.data.objects[o.name]=o
class LibraryAppendRegression(unittest.TestCase):
 def fixture(self,names):
  scene=SimpleNamespace(objects={},collection=SimpleNamespace(objects=None));linked=[]
  def link(o):linked.append(o);scene.objects[o.name]=o
  scene.collection.objects=SimpleNamespace(link=link);data=SimpleNamespace(objects={},libraries=None);context=LoadContext(names,data);data.libraries=SimpleNamespace(load=lambda *a,**k:context)
  return scene,SimpleNamespace(data=data),context,linked
 def test_mutating_library_list_does_not_mutate_caller_or_result_string_keys(self):
  names=['left_wrist_module_root','left_forearm_natural_hand'];original=names.copy();scene,bpy,context,linked=self.fixture(names)
  with patch.dict(sys.modules,{'bpy':bpy}):result=m._append_objects(scene,'data/action4_scene_assets/modular_wrist_v001/build_v003/left.blend',names)
  self.assertEqual(names,original);self.assertEqual(list(result),original);self.assertTrue(all(isinstance(k,str) for k in result));self.assertIsNot(context.assigned_list,names);self.assertEqual(len(linked),2);self.assertTrue(all(isinstance(o,FakeObject) for o in context.assigned_list))
 def test_immutable_tuple_source_names_with_explicit_instance_rename(self):
  names=('brace_module_root','brace_band');renames={'brace_module_root':'left_brace_module_root','brace_band':'left_brace_band'};scene,bpy,context,linked=self.fixture(names)
  with patch.dict(sys.modules,{'bpy':bpy}):result=m._append_objects(scene,'data/action4_scene_assets/modular_left_brace_v001/build_v001/brace_module_left.blend',names,renames)
  self.assertEqual(tuple(result),names);self.assertEqual([o.name for o in linked],list(renames.values()));self.assertEqual(names,('brace_module_root','brace_band'));self.assertTrue(isinstance(context.assigned_list,list))
if __name__=='__main__':unittest.main(verbosity=2)
