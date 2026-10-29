"""Phrase2 pixel orchestration adapted from the validated whole-frame backend.

Static adaptation of action4_global_render.execute_blender's target-dict loop.
Projection, canonical8, isolation, crop and measurements remain the existing
helpers. Import does not initialize Blender, dispatch workers or accept data.
"""
from pathlib import Path
import hashlib,json,re,traceback,numpy as np
from data import action4_global_render as render_helpers
from data import action4_global_environment as env
from data import action4_modular_render as modular_camera
from data import action4_render as base
from data import action4_synthesis as synth
from data import action4_phrase2_amodal as amodal
from data import action4_canonical_export as canonical_export
from data.action4_adapter import DataAdapter,_validated_png_size
ROOT=Path('/data/zy/vln/a-test')
REQUIRED_SOURCE_PATHS=('src/data/action4_phrase2_pixels.py','src/data/action4_global_render.py',
 'src/data/action4_global_environment.py','src/data/action4_render.py','src/data/action4_synthesis.py',
 'src/data/action4_amodal.py','src/data/action4_canonical_export.py','src/data/action4_adapter.py',
 'src/data/action4_modular_render.py','src/data/action4_mask_semantics.py','src/data/action4_phrase2_amodal.py')

def _bounded(value):
 p=Path(value);p=(p if p.is_absolute() else ROOT/p).resolve()
 if not p.is_relative_to(ROOT):raise ValueError('Project boundary')
 return p

def _sha(value):return hashlib.sha256(_bounded(value).read_bytes()).hexdigest()

def _write(path,value):
 with _bounded(path).open('x',encoding='utf-8') as file:json.dump(value,file,indent=2,allow_nan=False)

_VERIFIED_SOURCES={}
def _verify_sources(identities):
 if not isinstance(identities,dict) or not set(REQUIRED_SOURCE_PATHS).issubset(identities):raise ValueError('Complete pixel sources required')
 for name,digest in identities.items():
  if not isinstance(name,str) or Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Relative source required')
  p=_bounded(name);stat=p.stat();key=(digest,stat.st_size,stat.st_mtime_ns,stat.st_ino)
  if _VERIFIED_SOURCES.get(name)!=key:
   if _sha(name)!=digest:raise ValueError('Pixel source changed: '+name)
   _VERIFIED_SOURCES[name]=key

# Frozen project object-role names, independent of possibly wrong pass_index.
KNOWN_TARGET_PARTS={j:{'core':side+'_Action4_core','decal':side+'_Action4_core_top_id_decal'} for side,j in (('left',1),('right',2))}

def _known_target_role(name):
 for index,roles in KNOWN_TARGET_PARTS.items():
  for role,base_name in roles.items():
   if name==base_name or re.fullmatch(re.escape(base_name)+r'\.\d+',name):return index,role
 return None

def _camera_render_enabled(scene,obj):
 # Viewport hiding is not camera-render hiding. Inspect actual render controls.
 if obj.hide_render:return False
 visible=getattr(obj,'visible_camera',None)
 if visible is None:visible=getattr(getattr(obj,'cycles_visibility',None),'camera',None)
 if visible is None:raise ValueError('Known target needs explicit camera-ray visibility state')
 if not visible:return False
 layers=getattr(scene,'view_layers',None)
 if layers is not None and not any(getattr(layer,'use',True) and obj in set(layer.objects) for layer in layers):return False
 def visible_collection(collection):
  if collection.hide_render:return False
  return obj in set(collection.objects) or any(visible_collection(child) for child in collection.children)
 return visible_collection(scene.collection)

def _validate_targets(scene,entry,targets):
 if not isinstance(targets,dict) or any(type(j) is not int or j not in (1,2) for j in targets):raise ValueError('Physical target dict keys must be ID1/2')
 expected={j for j in synth.target_indices(entry) if j}
 if set(targets)!=expected:raise ValueError('Target dict differs from frozen left/right presence; right-only remains ID2')
 owned=set();scene_objects=set(scene.objects)
 for j,target in targets.items():
  if not isinstance(target,dict) or not {'core','parts'}.issubset(target):raise ValueError('Target core/parts required')
  core,parts=target['core'],target['parts']
  if not isinstance(parts,(list,tuple)) or not parts or core not in parts or len(set(parts))!=len(parts):raise ValueError('Distinct target parts including core required')
  if core.name!=KNOWN_TARGET_PARTS[j]['core'] or {p.name for p in parts}!=set(KNOWN_TARGET_PARTS[j].values()):raise ValueError('Present target source core/decal role mismatch')
  closure={core,*core.children_recursive}
  if any(p not in closure or p not in scene_objects or p.type!='MESH' or p.pass_index!=j for p in parts):raise ValueError('Target part ancestry/index mismatch')
  if {p for p in closure if p.type=='MESH' and p.pass_index==j}!=set(parts) or owned.intersection(parts):raise ValueError('Incomplete or overlapping target ownership')
  if any(not _camera_render_enabled(scene,p) for p in parts):raise ValueError('Present target core/decal cannot render to camera')
  owned.update(parts)
 for obj in scene_objects:
  role=_known_target_role(obj.name)
  if role is None:continue
  if obj.type!='MESH':raise ValueError('Known target source role must be mesh')
  if _camera_render_enabled(scene,obj):
   if role[0] not in expected:raise ValueError('Known absent target core/decal still participates in camera render, even with pass0')
   if obj not in owned:raise ValueError('Known camera-rendering target is not owned by expected source target parts')
 if any(o.pass_index!=0 for o in scene.objects if o not in owned):raise ValueError('Non-target carries target index')
 return sorted(expected)

def _paths(data_out,runtime_out,sid,present_ids,output_role):
 data,runtime=_bounded(data_out),_bounded(runtime_out)
 if output_role not in ('diagnostic','dataset_candidate'):raise ValueError('Explicit diagnostic or dataset_candidate output role required')
 data_area=ROOT/('outputs' if output_role=='diagnostic' else 'data')
 if data==data_area or not data.is_relative_to(data_area) or runtime==ROOT/'outputs' or not runtime.is_relative_to(ROOT/'outputs'):raise ValueError('Output directories disagree with explicit output role')
 if not data.is_dir() or not runtime.is_dir():raise ValueError('Caller-owned existing shard data/runtime directories required')
 for folder in ('rgb','masks'):
  p=_bounded(data/folder)
  if not p.is_relative_to(data):raise ValueError('Shard child directory escapes data output')
  p.mkdir(exist_ok=True)
 rgb=_bounded(data/'rgb'/(sid+'.png'));sample=_bounded(runtime/'samples'/sid)
 if not sample.is_relative_to(runtime) or not rgb.is_relative_to(data):raise ValueError('Sample output escapes assigned shard')
 # Both compositor nodes may write files even for a sparse right-only target.
 scene_file=_bounded(data/'scenes'/(sid+'.blend'))
 if not scene_file.is_relative_to(data):raise ValueError('Sample scene path escapes assigned data shard')
 collisions=[rgb,scene_file,sample]
 collisions.extend((data/'masks').glob(sid+'_i*'))
 if any(p.exists() or p.is_symlink() for p in collisions):raise FileExistsError('Per-sample artifacts are immutable; no automatic overwrite/retry')
 sample.parent.mkdir(exist_ok=True);sample.mkdir(exist_ok=False)
 return data,sample,rgb

def _canonical_geometry(**trusted_context):
 """Pure metadata only: original exporter and verifier, no observer lookup/math copy."""
 value=canonical_export.build_export(**trusted_context)
 canonical_export.verify_export(value,**trusted_context)
 return value

def _evaluated_world_matrix(core):
 import bpy
 return [list(row) for row in core.evaluated_get(bpy.context.evaluated_depsgraph_get()).matrix_world]

def _require_active_scene(scene):
 import bpy
 if bpy.context.scene is not scene:raise ValueError('Existing pixel helpers require the supplied scene to be active; no automatic context switch')

def _render_call(scene,write_still):
 import bpy
 bpy.ops.render.render(write_still=write_still)

def validate_host_pngs(record):
 """Host-only complete PNG decoding, dimensions and immutable child SHA checks."""
 if 'bpy' in __import__('sys').modules:raise RuntimeError('Full PNG validation belongs in the host Python runtime')
 if record.get('pending_host_png_validation') is not True or record.get('dataset_generation_allowed') is not False or record.get('training_accepted') is not False:raise ValueError('Unaccepted child record pending host PNG validation required')
 width,height=record['width'],record['height'];files=[]
 def verify_file(name,digest,size,role):
  if not isinstance(name,str) or Path(name).is_absolute() or '..' in Path(name).parts or not isinstance(digest,str) or not re.fullmatch('[0-9a-f]{64}',digest):raise ValueError('Exact relative image path and child SHA256 required')
  path=_bounded(name)
  if _sha(path)!=digest:raise ValueError('Child RGB/mask SHA256 mismatch: '+name)
  # The DataAdapter default check_image=True validates and fully decodes RGB.
  if role!='rgb' and _validated_png_size(path)!=tuple(size):raise ValueError('Mask PNG dimensions disagree: '+name)
  files.append({'path':name,'sha256':digest,'size_wh':list(size),'role':role})
 verify_file(record['image'],record['rgb_sha256'],[width,height],'rgb')
 for row in [*record['instances'],*record['excluded_instances']]:
  if row['amodal_expanded_shape_hw']!=[3*height,3*width]:raise ValueError('Original expanded amodal dimensions required')
  verify_file(row['mask_visible'],row['mask_visible_sha256'],[width,height],'visible')
  verify_file(row['mask_amodal_expanded'],row['mask_amodal_expanded_sha256'],[3*width,3*height],'amodal_expanded')
 DataAdapter(ROOT).validate_sample(record)
 from data import action4_mask_semantics as mask_semantics
 semantics=mask_semantics.verify_record_masks(record)
 for item in files:
  if _sha(item['path'])!=item['sha256']:raise ValueError('RGB/mask changed during host PNG validation')
 return {'status':'HOST_PNG_VALIDATION_PASS','PNG_decoded_and_dimensions_verified':True,'file_SHA256_verified':True,'mask_measurements_recomputed':True,'spatial_nesting_verified':True,'mask_semantics':semantics,'files':files,'dataset_generation_allowed':False,'training_accepted':False}

def render_frame(*,scene,entry,targets,canonical,K,data_out,runtime_out,output_role,
                 source_identities,render_settings,verify_scene,probe_only=False):
 """Render one upstream-authorized frame, return candidate record or probe geometry.

 output_role is required: diagnostic writes images and runtime under outputs;
 dataset_candidate writes images under data and runtime under outputs. Both
 remain not accepted. Caller owns these existing directories. Source
 identities are root-relative paths to SHA256, including REQUIRED_SOURCE_PATHS
 and canonical.source_manifest. verify_scene(scene) must be a read-only actual
 scene-contract check; it is always called before and after temporary work.
 Non-probe results remain pending_host_png_validation until the host calls
 validate_host_pngs with real PNG/SHA checks. No task launch or acceptance.
 """
 if not isinstance(entry,dict) or not callable(verify_scene) or type(probe_only) is not bool:raise ValueError('Explicit entry, verification callback and probe mode required')
 for key in ('sample_id','group_id','background_source_id','scene_family'):
  if not isinstance(entry.get(key),str) or not entry[key]:raise ValueError('Missing sample/group/family identity')
 if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,95}',entry['sample_id']) or entry.get('split') not in ('train','eval','test'):raise ValueError('Safe frozen sample ID and split required')
 if any(type(entry.get(key)) is not int or entry[key]<0 for key in ('index','seed','background_seed')):raise ValueError('Frozen integer sample/background seeds and index required')
 json.dumps(entry,allow_nan=False)
 ids=_validate_targets(scene,entry,targets)
 if render_settings.get('samples') not in (32,64) or render_settings.get('mask_samples')!=1:raise ValueError('Validated phrase2 RGB32/64 and visible1 policy required')
 amodal.validate_settings(render_settings);synth._denoiser_name(render_settings)
 k=np.asarray(K,float)
 if k.shape!=(3,3) or not np.isfinite(k).all():raise ValueError('Finite3x3 actual K required')
 actual_k=modular_camera.actual_K(scene)
 if not np.allclose(actual_k,k,atol=1e-6,rtol=0):raise ValueError('Supplied K differs from actual capture camera')
 mm,corners=canonical_export._canonical(canonical)
 source=canonical['source_manifest']
 _verify_sources(source_identities)
 if source_identities.get(source['path'])!=source['sha256']:raise ValueError('Canonical manifest must have explicit source SHA binding')
 asset=json.loads(_bounded(source['path']).read_text())
 width,height=scene.render.resolution_x,scene.render.resolution_y
 if type(width) is not int or type(height) is not int or min(width,height)<1 or scene.render.resolution_percentage!=100:raise ValueError('Explicit native frame dimensions at100percent required')
 data,sample,rgb=_paths(data_out,runtime_out,entry['sample_id'],ids,output_role);sid=entry['sample_id']
 _write(sample/'started.json',{'sample_id':sid,'probe_only':probe_only,'output_role':output_role,'planned_target_ids':ids,'source_identities':source_identities,'automatic_retry':False,'dataset_generation_allowed':False})
 before=None;calls=0;attempts=0;primary=None;restoration=[];exports={};projections={};coverage={};instances=[];excluded=[];record=None
 try:
  before=render_helpers.state_snapshot(scene)
  _require_active_scene(scene)
  verify_scene(scene)
  with render_helpers.temporary_render_state(scene):
   scene.render.threads_mode='FIXED';scene.render.threads=2;scene.frame_set(1)
   scene.render.resolution_x=width;scene.render.resolution_y=height;scene.render.resolution_percentage=100
   scene.render.use_border=False;scene.render.use_crop_to_border=False;scene.render.film_transparent=False
   scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB';scene.render.image_settings.color_depth='8'
   scene.cycles.samples=render_settings['samples'];scene.cycles.seed=entry['seed'];scene.cycles.max_bounces=6
   scene.cycles.diffuse_bounces=2;scene.cycles.glossy_bounces=2;scene.cycles.transparent_max_bounces=4;scene.cycles.use_adaptive_sampling=False
   inspection=render_helpers.inspection_visibility_guard(scene,targets,corners,[])
   # Static target-dict pattern from global_render: all original8 exports before isolation.
   for j in ids:
    target=targets[j];points,visible,rays=render_helpers.project_canonical(scene,target['core'],corners,return_ray_clear=True)
    exports[j]=_canonical_geometry(canonical=canonical,instance_id=j,object_to_world_m=_evaluated_world_matrix(target['core']),camera_to_world_blender_m=[list(row) for row in scene.camera.matrix_world],K=actual_k,image_size_wh=[width,height],near_plane_m=scene.camera.data.clip_start,ray_clear_to_virtual_point=rays,legacy_corners_px=points,legacy_corner_visible=visible);projections[j]=(points,visible)
    with render_helpers.isolated_expanded(scene,target['parts']):coverage[j]=amodal.geometry_preflight(scene,target['core'],target['parts'],corners)
   if not probe_only:
    device=synth._render_device(scene,render_settings);denoiser=synth._set_denoiser(scene,render_settings)
    nodes=base._mask_nodes(data/'masks') if ids else []
    for node in nodes:node.mute=True
    scene.cycles.samples=render_settings['samples'];scene.cycles.use_denoising=True;scene.render.filepath=str(rgb)
    attempts+=1;_render_call(scene,True);calls+=1
    if ids:
     for j,node in enumerate(nodes,1):node.mute=False;node.file_slots[0].path=f'{sid}_i{j}_visible_'
     scene.cycles.samples=1;scene.cycles.use_denoising=False
     attempts+=1;_render_call(scene,False);calls+=1
    for j in ids:
     target=targets[j];vp=data/'masks'/f'{sid}_i{j}_visible_0001.png';ap=data/'masks'/f'{sid}_i{j}_amodal_for_{j}_0001.png'
     for n,node in enumerate(nodes,1):node.file_slots[0].path=f'{sid}_i{n}_amodal_for_{j}_'
     with render_helpers.isolated_expanded(scene,target['parts']):
      attempts+=1
      if render_settings.get('amodal_mode','full')=='crop':full,info=amodal.render_crop(scene,scene.camera,target['core'],target['parts'],corners,nodes,j,ap,sample/'amodal_crop'/('i'+str(j)))
      else:_render_call(scene,False);full=base._mask_array(ap);info={'mode':'full'}
      calls+=1
     measurements=render_helpers.mask_measurements(base._mask_array(vp),full,width,height)
     common={'instance_id':j,'object_id':1,'target_parts':[p.name for p in target['parts']],
       'canonical8_geometry':exports[j],'object_to_world_m':exports[j]['object_to_world_m'],
       'corners_px':projections[j][0],'corner_visible':projections[j][1],
       'mask_visible':str(vp.relative_to(ROOT)),'mask_amodal_expanded':str(ap.relative_to(ROOT)),
       'mask_visible_sha256':_sha(vp),'mask_amodal_expanded_sha256':_sha(ap),
       'amodal_expanded_origin_px':[-width,-height],'amodal_expanded_shape_hw':[3*height,3*width],'amodal_render':info}
     if measurements.pop('excluded',False):excluded.append({**common,**measurements});continue
     points=np.asarray(projections[j][0]);instances.append({**common,'bbox_full_xyxy':[float(points[:,0].min()),float(points[:,1].min()),float(points[:,0].max()),float(points[:,1].max())],**measurements})
    record={**entry,'schema_version':'action4.v1','image':str(rgb.relative_to(ROOT)),'width':width,'height':height,
      'instances':instances,'excluded_instances':excluded,'target_count':len(instances),'planned_target_count':entry['target_count'],
      'corners_3d_mm':mm.tolist(),'canonical_source_manifest':source,'asset_version':asset.get('asset_version'),
      'source_stl_sha256':asset.get('source_sha256'),'camera':{'K':k.tolist(),'camera_to_world_m':[list(row) for row in scene.camera.matrix_world],'pixel_convention':'integer_centers','calibration_status':'upstream scene prior; not calibrated by pixel helper'},
      'render_settings':{**render_settings,'actual_device':device,'actual_denoiser':denoiser},'inspection_visibility':inspection,
      'amodal_geometry_coverage':coverage,'synthesis_code_sha256':source_identities['src/data/action4_synthesis.py'],
      'render_base_sha256':source_identities['src/data/action4_render.py'],'pixel_helper_sha256':source_identities['src/data/action4_phrase2_pixels.py'],
      'pixel_source_identities':source_identities,'rgb_sha256':_sha(rgb),'output_role':output_role,'status':'diagnostic_render_not_training_accepted' if output_role=='diagnostic' else 'render_candidate_not_training_accepted','dataset_generation_allowed':False,'training_accepted':False}
    record['pending_host_png_validation']=True;record['structure_semantics_validated']=True
    DataAdapter(ROOT).validate_sample(record,check_image=False)
 except Exception as exc:primary=(exc,traceback.format_exc())
 finally:
  try:
   if before is not None and render_helpers.state_snapshot(scene)!=before:restoration.append('Temporary render/camera/hide state differs')
  except Exception as exc:restoration.append('Render-state verification: '+str(exc))
  try:_validate_targets(scene,entry,targets)
  except Exception as exc:restoration.append('Target presence/camera eligibility verification: '+str(exc))
  try:verify_scene(scene)
  except Exception as exc:restoration.append('Scene verification callback: '+str(exc))
  try:_verify_sources(source_identities)
  except Exception as exc:restoration.append('Source verification: '+str(exc))
 if primary or restoration:
  _write(sample/'failure.json',{'status':'FAILED_RETAINED_NO_RETRY','sample_id':sid,'output_role':output_role,'primary_error':str(primary[0]) if primary else None,'primary_traceback':primary[1] if primary else None,'restoration_errors':restoration,'render_calls_completed':calls,'render_attempts':attempts,'dataset_generation_allowed':False})
  if primary:raise primary[0]
  raise RuntimeError('Pixel restoration/source check failed: '+str(restoration))
 result={'status':'PIXEL_GEOMETRY_PROBE_ONLY' if probe_only else 'PIXEL_RENDER_COMPLETE_PENDING_HOST_PNG_VALIDATION','record':record,'output_role':output_role,'geometry':exports,'geometry_coverage':coverage,'rendered':not probe_only,'render_calls':calls,'render_attempts':attempts,'restoration_verified':True,'source_identities_verified':True,'dataset_generation_allowed':False,'training_accepted':False}
 result['pending_host_png_validation']=not probe_only
 _write(sample/('probe.json' if probe_only else 'record.json'),record if not probe_only else result)
 if not probe_only:
  result['record_ref']={'path':str((sample/'record.json').relative_to(ROOT)),'sha256':_sha(sample/'record.json')}
  _write(sample/'render_receipt.json',result)
 return result
