"""CPU mock orchestration, real canonical math/mask accounting/DataAdapter. No bpy/GPU."""
from pathlib import Path
import sys,unittest,importlib.util,tempfile,copy,json,hashlib,inspect
from contextlib import contextmanager,ExitStack
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
from PIL import Image
R=Path('/data/zy/vln/a-test');O=R/'outputs/action4_modular_pixels_prepare_v004';sys.dont_write_bytecode=True;sys.path.insert(0,str(R/'src'))
spec=importlib.util.find_spec('data.action4_modular_pixels');pixel=__import__('data.action4_modular_pixels',fromlist=['render_frame']) if spec else None
from data import action4_canonical_export as canonical_math
from data import action4_mask_semantics as mask_semantics
from data.action4_adapter import DataAdapter
class Obj:
 def __init__(self,name,index):
  self.name=name;self.type='MESH';self.pass_index=index;self.hide_render=False;self.visible_camera=True;self.hide_viewport=False;self.children_recursive=[];self.matrix_world=np.eye(4);self.matrix_world[:3,3]=[-.15 if index==1 else .15,0,-1]
class SceneObjects(list):
 def __getitem__(self,key):
  if isinstance(key,str):
   for obj in self:
    if obj.name==key:return obj
   raise KeyError(key)
  return super().__getitem__(key)
class Scene:
 def __init__(self,objects):
  self.objects=SceneObjects(objects);self.collection=NS(hide_render=False,objects=self.objects,children=[]);self.frame_current=9;self.render=NS(resolution_x=32,resolution_y=24,resolution_percentage=100,pixel_aspect_x=1,pixel_aspect_y=1,image_settings=NS(file_format='PNG',color_mode='RGB',color_depth='8'),filepath='original.png');self.cycles=NS(samples=19,use_denoising=False,seed=3);self.camera=NS(name='assembly_observer',matrix_world=np.eye(4),data=NS(type='PERSP',sensor_fit='HORIZONTAL',shift_x=0,shift_y=0,lens=40,clip_start=.001,sensor_width=32))
 def frame_set(self,n):self.frame_current=n
@contextmanager
def harness(case,ids,*,excluded=(),fail_id=None,output_role='dataset_candidate'):
 case.assertIsNotNone(pixel,'new modular pixel helper is not implemented')
 with tempfile.TemporaryDirectory(dir=O) as temp:
  root=Path(temp);data=root/('outputs/diagnostic_run/pixels' if output_role=='diagnostic' else 'data/shard');run=root/('outputs/diagnostic_run/runtime' if output_role=='diagnostic' else 'outputs/shard');(data/'rgb').mkdir(parents=True);(data/'masks').mkdir();run.mkdir(parents=True)
  targets={i:{'core':Obj(('left' if i==1 else 'right')+'_Action4_core',i),'parts':[]} for i in ids}
  for i,t in targets.items():
   decal=Obj(('left' if i==1 else 'right')+'_Action4_core_top_id_decal',i);t['core'].children_recursive=[decal];t['parts']=[t['core'],decal]
   if i in excluded:t['core'].matrix_world[0,3]=.9
  scene=Scene([part for t in targets.values() for part in t['parts']]);K=[[40,0,15.5],[0,40,11.5],[0,0,1]]
  corners=[[-40,-20,-30],[40,-20,-30],[40,20,-30],[-40,20,-30],[-40,-20,30],[40,-20,30],[40,20,30],[-40,20,30]]
  manifest=root/'data/cad.json';manifest.parent.mkdir(exist_ok=True);manifest.write_text(json.dumps({'asset_version':'CPU_MOCK_ONLY','source_sha256':'f'*64}));mh=hashlib.sha256(manifest.read_bytes()).hexdigest()
  canon={'source_manifest':{'path':'data/cad.json','sha256':mh},'corner_order':list(canonical_math.ORDER),'corners_3d_mm':corners,'corners_3d_m':(np.array(corners)*.001).tolist()}
  entry={'sample_id':'mock_frame','index':0,'seed':7,'scene_family':'hard_negative' if not ids else 'wall_rear','target_count':len(ids),'target_presence':[1 in ids,2 in ids],'group_id':'mock_group','background_source_id':'mock_background','background_seed':8,'split':'train','status':'training_accepted'}
  settings={'device':'CUDA','worker_gpu_uuid':'CPU_MOCK','denoiser':'OPTIX','samples':64,'mask_samples':1,'amodal_mode':'crop','amodal_margin_px':16}
  sources={name:'a'*64 for name in pixel.REQUIRED_SOURCE_PATHS};sources['data/cad.json']=mh
  calls={'render':[],'amodal':[],'verify':0};nodes=[NS(mute=False,file_slots=[NS(path='')]) for _ in range(2)]
  def state(s):return copy.deepcopy({'render':vars(s.render),'cycles':vars(s.cycles),'camera':vars(s.camera.data),'frame':s.frame_current,'hide':[(x.hide_render,x.hide_viewport) for x in s.objects]})
  @contextmanager
  def temporary(s):
   old=state(s)
   try:yield
   finally:
    for target,values in [(s.render,old['render']),(s.cycles,old['cycles']),(s.camera.data,old['camera'])]:target.__dict__.clear();target.__dict__.update(copy.deepcopy(values))
    s.frame_current=old['frame']
    for x,hidden in zip(s.objects,old['hide']):x.hide_render,x.hide_viewport=hidden
  @contextmanager
  def isolate(s,parts):
   width,height=s.render.resolution_x,s.render.resolution_y;sw=s.camera.data.sensor_width;hidden=[x.hide_render for x in s.objects]
   try:
    s.render.resolution_x*=3;s.render.resolution_y*=3;s.camera.data.sensor_width*=3
    for x in s.objects:x.hide_render=x not in parts
    yield
   finally:
    s.render.resolution_x=width;s.render.resolution_y=height;s.camera.data.sensor_width=sw
    for x,h in zip(s.objects,hidden):x.hide_render=h
  def export(s,core,c,j,points,visible,rays):
   context={'canonical':c,'instance_id':j,'object_to_world_m':core.matrix_world.tolist(),'camera_to_world_blender_m':s.camera.matrix_world.tolist(),'K':K,'image_size_wh':[32,24],'near_plane_m':.001,'ray_clear_to_virtual_point':rays,'legacy_corners_px':points,'legacy_corner_visible':visible};v=canonical_math.build_export(**context);canonical_math.verify_export(v,**context);return v
  def project(s,core,corners,return_ray_clear=False):
   v=canonical_math.build_export(canonical=canon,instance_id=core.pass_index,object_to_world_m=core.matrix_world.tolist(),camera_to_world_blender_m=s.camera.matrix_world.tolist(),K=K,image_size_wh=[32,24],near_plane_m=.001);return v['points_native_px'],v['point_inframe'],[True]*8
  def visible(j):
   a=np.zeros((24,32),bool)
   if j in ids and j not in excluded:a[9:13,4:8 if j==1 else 8]=True if j==1 else False
   if j==2 and j in ids and j not in excluded:a[9:13,21:25]=True
   return a
  def render(s,write_still):
   calls['render'].append((write_still,s.cycles.samples))
   if write_still:Image.new('RGB',(32,24),'gray').save(s.render.filepath)
   else:
    for j,node in enumerate(nodes,1):Image.fromarray(visible(j).astype('uint8')*255).save(data/'masks'/(node.file_slots[0].path+'0001.png'))
  def crop(s,camera,core,parts,corners,nodes,j,ap,scratch):
   calls['amodal'].append(j)
   if j==fail_id:raise RuntimeError('second-amodal-failure')
   full=np.zeros((72,96),bool)
   if j in excluded:full[0:4,0:4]=True
   else:full[24:48,32:64]=visible(j)
   Image.fromarray(full.astype('uint8')*255).save(ap);return full,{'mode':'crop','CPU_MOCK_ONLY':True}
  def verify(s):calls['verify']+=1
  patches=[patch.object(pixel,'ROOT',root),patch.object(mask_semantics,'ROOT',root),patch.object(pixel,'_require_active_scene',return_value=None),patch.object(pixel,'_verify_sources',return_value=None),patch.object(pixel.render_helpers,'state_snapshot',side_effect=state),patch.object(pixel.render_helpers,'temporary_render_state',side_effect=temporary),patch.object(pixel.render_helpers,'isolated_expanded',side_effect=isolate),patch.object(pixel.render_helpers,'inspection_visibility_guard',return_value={'CPU_MOCK_ONLY':True}),patch.object(pixel.render_helpers,'project_canonical',side_effect=project),patch.object(pixel.render_helpers,'crop_geometry_preflight',return_value={'CPU_MOCK_ONLY':True}),patch.object(pixel.base,'_mask_nodes',return_value=nodes),patch.object(pixel.base,'_mask_array',side_effect=lambda p:np.array(Image.open(p))>127),patch.object(pixel.synth,'_render_device',return_value={'CPU_MOCK_ONLY':True}),patch.object(pixel.synth,'_set_denoiser',return_value='CPU_MOCK_ONLY'),patch.object(pixel.amodal,'render_crop',side_effect=crop),patch.object(pixel,'_render_call',side_effect=render)]
  if hasattr(pixel,'_evaluated_world_matrix'):patches.append(patch.object(pixel,'_evaluated_world_matrix',side_effect=lambda core:core.matrix_world.tolist()))
  initial=state(scene)
  with ExitStack() as stack:
   for item in patches:stack.enter_context(item)
   args={'scene':scene,'entry':entry,'targets':targets,'canonical':canon,'K':K,'data_out':data,'runtime_out':run,'output_role':output_role,'source_identities':sources,'render_settings':settings,'verify_scene':verify}
   yield NS(args=args,scene=scene,calls=calls,data=data,run=run,root=root,initial=initial,state=state)
class PixelTests(unittest.TestCase):
 def test_zero_targets_rgb_once_no_masks(self):
  with harness(self,[]) as h:
   result=pixel.render_frame(**h.args);self.assertEqual(h.calls['render'],[(True,64)]);self.assertEqual(h.calls['amodal'],[]);self.assertEqual(result['record']['instances'],[]);self.assertFalse(result['record']['dataset_generation_allowed']);DataAdapter(h.root).validate_sample(result['record'])
 def test_right_only_keeps_ID2_and_compatible_record(self):
  with harness(self,[2]) as h:
   result=pixel.render_frame(**h.args);r=result['record'];self.assertEqual(h.calls['render'],[(True,64),(False,1)]);self.assertEqual(h.calls['amodal'],[2]);self.assertEqual([x['instance_id'] for x in r['instances']],[2]);self.assertEqual(r['status'],'render_candidate_not_training_accepted');self.assertEqual(r['synthesis_code_sha256'],'a'*64);DataAdapter(h.root).validate_sample(r);self.assertEqual(h.state(h.scene),h.initial)
 def test_dual_once_then_each_amodal(self):
  with harness(self,[1,2]) as h:
   r=pixel.render_frame(**h.args);self.assertEqual(h.calls['render'],[(True,64),(False,1)]);self.assertEqual(h.calls['amodal'],[1,2]);self.assertEqual(r['render_calls'],4);self.assertEqual([x['instance_id'] for x in r['record']['instances']],[1,2]);DataAdapter(h.root).validate_sample(r['record'])
 def test_excluded_retains_original8(self):
  with harness(self,[2],excluded=[2]) as h:
   r=pixel.render_frame(**h.args)['record'];self.assertEqual(r['instances'],[]);self.assertEqual(r['planned_target_count'],1);self.assertEqual(r['target_count'],0);self.assertEqual(r['target_presence'],[False,True]);self.assertNotEqual(len(r['instances']),h.args['entry']['target_count']);self.assertEqual(len(r['excluded_instances'][0]['canonical8_geometry']['points_object_mm']),8);self.assertEqual(r['excluded_instances'][0]['reason'],'outside_image')
 def test_second_amodal_failure_restores_and_refuses_retry(self):
  with harness(self,[1,2],fail_id=2) as h:
   with self.assertRaisesRegex(RuntimeError,'second-amodal-failure'):pixel.render_frame(**h.args)
   self.assertEqual(h.state(h.scene),h.initial);self.assertEqual(h.calls['verify'],2);self.assertTrue((h.run/'samples/mock_frame/failure.json').is_file());self.assertFalse((h.run/'samples/mock_frame/record.json').exists())
   count=len(h.calls['render'])
   with self.assertRaises(FileExistsError):pixel.render_frame(**h.args)
   self.assertEqual(len(h.calls['render']),count)
 def test_probe_has_no_RGB_or_candidate_record(self):
  with harness(self,[2]) as h:
   h.args['probe_only']=True;r=pixel.render_frame(**h.args)
   self.assertFalse(r['rendered']);self.assertIsNone(r['record']);self.assertEqual(h.calls['render'],[]);self.assertEqual(h.calls['amodal'],[]);self.assertEqual(len(r['geometry'][2]['points_object_mm']),8);self.assertEqual(h.state(h.scene),h.initial)
 def test_final_scene_callback_failure_cannot_return_success(self):
  with harness(self,[2]) as h:
   def verify(s):
    h.calls['verify']+=1
    if h.calls['verify']==2:raise ValueError('actual-scene-contract-drift')
   h.args['verify_scene']=verify
   with self.assertRaisesRegex(RuntimeError,'actual-scene-contract-drift'):pixel.render_frame(**h.args)
   failure=json.loads((h.run/'samples/mock_frame/failure.json').read_text());self.assertIsNone(failure['primary_error']);self.assertIn('actual-scene-contract-drift',str(failure['restoration_errors']));self.assertFalse((h.run/'samples/mock_frame/record.json').exists())
 def test_absent_known_visible_core_or_decal_rejected_even_with_pass0(self):
  for name in ['right_Action4_core','right_Action4_core_top_id_decal','right_Action4_core.001']:
   with self.subTest(name=name),harness(self,[]) as h:
    h.scene.objects.append(Obj(name,0))
    with self.assertRaisesRegex(ValueError,'absent|presence|known'):pixel.render_frame(**h.args)
    self.assertEqual(h.calls['render'],[])
 def test_explicitly_nonrendering_absent_known_parts_allowed(self):
  for flag in ['hide_render','visible_camera']:
   with self.subTest(flag=flag),harness(self,[]) as h:
    for name in ['right_Action4_core','right_Action4_core_top_id_decal']:
     obj=Obj(name,0);setattr(obj,flag,True if flag=='hide_render' else False);h.scene.objects.append(obj)
    r=pixel.render_frame(**h.args)['record'];self.assertEqual(r['target_presence'],[False,False]);self.assertEqual(r['instances'],[]);self.assertEqual(h.calls['render'],[(True,64)])
 def test_present_hidden_camera_invisible_or_wrong_source_role_rejected(self):
  for mode in ['hide_render','visible_camera','collection_hidden','wrong_side_role']:
   with self.subTest(mode=mode),harness(self,[2]) as h:
    core=h.args['targets'][2]['core']
    if mode=='hide_render':core.hide_render=True
    elif mode=='visible_camera':core.visible_camera=False
    elif mode=='collection_hidden':h.scene.collection.hide_render=True
    else:core.name='left_Action4_core'
    with self.assertRaises(ValueError):pixel.render_frame(**h.args)
    self.assertEqual(h.calls['render'],[])
 def test_output_roles_legal_and_cross_directory_misuse(self):
  self.assertIn('output_role',inspect.signature(pixel.render_frame).parameters,'Role must be an explicit required API keyword')
  self.assertIs(inspect.signature(pixel.render_frame).parameters['output_role'].default,inspect.Parameter.empty)
  for role in ['diagnostic','dataset_candidate']:
   with self.subTest(role=role),harness(self,[2],output_role=role) as h:
    r=pixel.render_frame(**h.args)['record'];self.assertEqual(r['output_role'],role);self.assertFalse(r['dataset_generation_allowed']);self.assertFalse(r['training_accepted']);self.assertEqual(h.calls['render'],[(True,64),(False,1)]);DataAdapter(h.root).validate_sample(r)
    self.assertEqual(r['status'],'diagnostic_render_not_training_accepted' if role=='diagnostic' else 'render_candidate_not_training_accepted')
  for role,wrong in [('dataset_candidate','diagnostic'),('diagnostic','dataset_candidate'),('diagnostic','bogus')]:
   with self.subTest(role=role,wrong=wrong),harness(self,[2],output_role=role) as h:
    h.args['output_role']=wrong
    with self.assertRaises(ValueError):pixel.render_frame(**h.args)
    self.assertEqual(h.calls['render'],[])
  with harness(self,[2],output_role='diagnostic') as h:
   wrong=h.root/'data/wrong_runtime';wrong.mkdir(parents=True);h.args['runtime_out']=wrong
   with self.assertRaises(ValueError):pixel.render_frame(**h.args)
   self.assertEqual(h.calls['render'],[])
 def test_real_assembly_camera_metadata_without_g1b_observer(self):
  with harness(self,[2]) as h:
   self.assertEqual(h.scene.camera.name,'assembly_observer')
   with self.assertRaises(KeyError):pixel.env.actual_K(h.scene)
   try:r=pixel.render_frame(**h.args)['record']
   except (KeyError,ModuleNotFoundError) as exc:self.fail('Real assembly camera selection/metadata still depends on G1 environment: '+str(exc))
   self.assertEqual(r['camera']['K'],[[40.,0.,15.5],[0.,40.,11.5],[0.,0.,1.]])
   self.assertEqual(r['instances'][0]['canonical8_geometry']['K_native'],r['camera']['K'])
 def test_pure_metadata_wrapper_uses_trusted_K_and_original_export(self):
  self.assertTrue(callable(getattr(pixel,'_canonical_geometry',None)),'Pure metadata adapter missing')
  with harness(self,[2]) as h:
   core=h.args['targets'][2]['core'];K=h.args['K'];context={'canonical':h.args['canonical'],'instance_id':2,'object_to_world_m':core.matrix_world.tolist(),'camera_to_world_blender_m':h.scene.camera.matrix_world.tolist(),'K':K,'image_size_wh':[32,24],'near_plane_m':.001}
   expected=canonical_math.build_export(**context);self.assertEqual(pixel._canonical_geometry(**context),expected)
   context['legacy_corners_px']=expected['points_native_px'];context['legacy_corner_visible']=expected['point_inframe'];context['ray_clear_to_virtual_point']=[True]*8
   value=pixel._canonical_geometry(**context);self.assertEqual(value['legacy_corners_px'],expected['points_native_px'])
   wrong=copy.deepcopy(context);wrong['K'][0][0]*=2
   with self.assertRaises(ValueError):pixel._canonical_geometry(**wrong)
 def test_wrong_ID_and_source_hash_early_rejected(self):
  with harness(self,[2]) as h:
   h.args['targets']={1:h.args['targets'][2]}
   with self.assertRaises(ValueError):pixel.render_frame(**h.args)
   self.assertEqual(h.calls['render'],[])
  self.assertIsNotNone(pixel)
  refs={name:hashlib.sha256((R/name).read_bytes()).hexdigest() for name in pixel.REQUIRED_SOURCE_PATHS};refs[next(iter(refs))]='0'*64
  with self.assertRaises(ValueError):pixel._verify_sources(refs)
  with patch.dict(sys.modules,{'bpy':NS(context=NS(scene=object()))}):
   with self.assertRaisesRegex(ValueError,'active'):pixel._require_active_scene(object())
 def test_existing_sample_artifact_refused_without_render(self):
  with harness(self,[2]) as h:
   (h.data/'rgb/mock_frame.png').write_bytes(b'owned-existing-artifact')
   with self.assertRaises(FileExistsError):pixel.render_frame(**h.args)
   self.assertEqual((h.data/'rgb/mock_frame.png').read_bytes(),b'owned-existing-artifact');self.assertEqual(h.calls['render'],[])
if __name__=='__main__':unittest.main(verbosity=2)
