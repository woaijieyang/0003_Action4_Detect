"""One root-approved bilateral diagnostic adapter. No scene/render on import.

The root supplies a frozen scene config in the unchanged bilateral.prepare
schema. Its validated output_scope owns each run and root_run_approval.json;
scene versions never require copying this adapter or relaxing pixel math.
"""
from pathlib import Path
import argparse,hashlib,importlib,json,os,re,subprocess,sys,traceback
from data import action4_modular_pixels as pixels
from data.action4_devices import ALLOWED_GPU_UUIDS,worker_environment
ROOT=Path('/data/zy/vln/a-test')
OWN=ROOT/'outputs/action4_modular_bilateral_preview_prepare_v004'
SCENE_MODULE=ROOT/'src/data/action4_modular_bilateral_scene.py'
ENTRYPOINT=ROOT/'scripts/action4_modular_bilateral_preview.py'
PIXELS_HANDOFF=ROOT/'outputs/action4_modular_pixels_prepare_v004/handoff.json'
PIXELS_SPEC={'path':'outputs/action4_modular_pixels_spec_v004/report.json','sha256':'d0f15c0070484850d57bca9bb771ccb1a31bb2745676a8f3f9d13c05f31e8545'}
CPU_STATUS='APPROVED_ONE_BILATERAL_CPU_PROBE'
GPU_STATUS='APPROVED_ONE_BILATERAL_GPU_DIAGNOSTIC'
APPROVAL_BINDING_KEYS=('scene_config','scene_source','scene_handoff','pixels_handoff','pixels_spec','adapter_handoff')
MOUNT_KEYS=('quick_release','usb_connector','white_module','ball','link','cable')

def bounded(value):
 p=Path(value)
 if '..' in p.parts:raise ValueError('Parent traversal forbidden')
 p=(p if p.is_absolute() else ROOT/p).resolve()
 if not p.is_relative_to(ROOT):raise ValueError('Project boundary')
 return p

def sha(p):return hashlib.sha256(bounded(p).read_bytes()).hexdigest()
def read(p):return json.loads(bounded(p).read_text())
def ref(p):
 p=bounded(p);return {'path':str(p.relative_to(ROOT)),'sha256':sha(p)}
def write(p,value):
 with bounded(p).open('x',encoding='utf-8') as file:json.dump(value,file,indent=2,allow_nan=False)
def check_ref(value):
 if not isinstance(value,dict) or set(value)!={'path','sha256'} or not isinstance(value['path'],str) or Path(value['path']).is_absolute() or not re.fullmatch('[0-9a-f]{64}',value['sha256']):raise ValueError('Exact relative path/SHA256 reference required')
 p=bounded(value['path'])
 if sha(p)!=value['sha256']:raise ValueError('Bound source/review changed: '+value['path'])
 return p

def run_root_from_config(c):
 runs=bounded(c['output_scope'])
 if runs.name!='runs' or not runs.is_relative_to(ROOT/'outputs') or runs.parent==ROOT/'outputs':raise ValueError('Scene-declared outputs subdirectory/runs required')
 return runs

def output_path(value,*,allowed_runs,fresh=True):
 p=bounded(value);runs=bounded(allowed_runs)
 if not runs.is_relative_to(ROOT/'outputs') or p==runs or not p.is_relative_to(runs):raise ValueError('Fresh child of the prepared scene output_scope required')
 if fresh and (p.exists() or p.is_symlink()):raise FileExistsError('Fresh single attempt; old failures/completions are immutable')
 return p

def resources(threads,gpu_uuid,probe_only):
 if type(threads) is not int or threads!=2 or type(probe_only) is not bool:raise ValueError('Explicit CPU2/probe mode required')
 if (probe_only and gpu_uuid is not None) or (not probe_only and gpu_uuid not in ALLOWED_GPU_UUIDS):raise ValueError('CPU probe has no GPU; GPU diagnostic requires one allowed UUID')

def build_command(config,out,binary,*,threads,gpu_uuid,probe_only):
 resources(threads,gpu_uuid,probe_only);scope=run_root_from_config(read(config));out=output_path(out,allowed_runs=scope,fresh=False)
 cmd=[str(bounded(binary)),'--factory-startup','--disable-autoexec','-noaudio','--background','--threads','2','--python-exit-code','1','--python',str(ENTRYPOINT),'--','--inside-blender','--config',str(bounded(config)),'--output',str(out),'--threads','2']
 return cmd+(['--probe-only'] if probe_only else ['--gpu-uuid',gpu_uuid])

def _handoff_files(value):
 if not isinstance(value.get('files'),dict) or not value['files']:raise ValueError('Sealed handoff file identities required')
 for p,h in value['files'].items():check_ref({'path':p,'sha256':h})

def _pixel_release():
 spec=read(check_ref(PIXELS_SPEC));hand=read(PIXELS_HANDOFF);_handoff_files(hand)
 if spec.get('status')!='PASS' or spec['handoff_sha256']!=sha(PIXELS_HANDOFF) or spec['source_sha256']!=sha(pixels.__file__):raise ValueError('Approved pixels release binding')
 pixels._verify_sources(hand['actual_source_refs'])
 return hand

def prepare(config):
 if Path.cwd().resolve()!=ROOT:raise ValueError('Physical project root')
 backend=importlib.import_module('data.action4_modular_bilateral_scene')
 if Path(backend.__file__).resolve()!=SCENE_MODULE:raise ValueError('Canonical scene module path required')
 c,ctx=backend.prepare(bounded(config));runs=run_root_from_config(c)
 if runs!=bounded(backend.OWN)/'runs':raise ValueError('Scene prepare/OWN/output_scope disagreement')
 if c['sources']['bilateral_module']!=ref(SCENE_MODULE):raise ValueError('Root-selected scene source differs from loaded canonical module')
 _pixel_release()
 return backend,c,ctx

def check_approval_fields(a,bindings,out,*,threads,gpu_uuid,probe_only):
 resources(threads,gpu_uuid,probe_only)
 if not isinstance(a,dict) or a.get('status')!=(CPU_STATUS if probe_only else GPU_STATUS) or type(a.get('samples')) is not int or a['samples']!=1 or type(a.get('threads')) is not int or a['threads']!=2 or a.get('probe_only') is not probe_only or a.get('gpu_uuid')!=gpu_uuid:raise ValueError('Exact single-view mode/resources approval required')
 if bounded(a.get('output_dir',''))!=out:raise ValueError('Approval output mismatch')
 if any(a.get(key)!=bindings[key] for key in APPROVAL_BINDING_KEYS):raise ValueError('Scene/config/source/handoff approval binding mismatch')
 reviews=a.get('reviews')
 if not isinstance(reviews,dict) or set(reviews)!={'spec_review','root_review'} or any(not isinstance(r,dict) or set(r)!={'path','sha256','pass'} or r['pass'] is not True for r in reviews.values()):raise ValueError('Explicit spec and root review PASS/hash bindings required')


def approval(config,c,out,*,threads,gpu_uuid,probe_only):
 runs=run_root_from_config(c);out=output_path(out,allowed_runs=runs,fresh=False);approval_path=runs.parent/'root_run_approval.json';a=read(approval_path)
 scene_hand_path=check_ref(a['scene_handoff'])
 if not scene_hand_path.is_relative_to(ROOT/'outputs'):raise ValueError('Scene handoff must be outputs evidence')
 scene_hand=read(scene_hand_path);_handoff_files(scene_hand)
 config_ref=ref(config);source_ref=ref(SCENE_MODULE)
 if scene_hand.get('source')!=source_ref or scene_hand.get('config')!=config_ref:raise ValueError('Final root-selected scene handoff must bind actual config/source')
 bindings={'scene_config':config_ref,'scene_source':source_ref,'scene_handoff':ref(scene_hand_path),'pixels_handoff':ref(PIXELS_HANDOFF),'pixels_spec':PIXELS_SPEC,'adapter_handoff':ref(OWN/'handoff.json')}
 check_approval_fields(a,bindings,out,threads=threads,gpu_uuid=gpu_uuid,probe_only=probe_only)
 for r in bindings.values():check_ref(r)
 _handoff_files(read(OWN/'handoff.json'))
 for r in a['reviews'].values():check_ref({k:r[k] for k in ('path','sha256')})
 return a,approval_path,bindings

def source_identities(config,c,ctx,bindings,approval_path):
 refs=dict(read(PIXELS_HANDOFF)['actual_source_refs'])
 def add(r):
  check_ref(r)
  if r['path'] in refs and refs[r['path']]!=r['sha256']:raise ValueError('Conflicting source identities')
  refs[r['path']]=r['sha256']
 for r in c['sources'].values():add(r)
 for conf in [ctx['right_config'],ctx['right_context']['neutral_config']]:
  for r in conf['sources'].values():add(r)
 for r in bindings.values():add(r)
 for r in read(approval_path)['reviews'].values():add({k:r[k] for k in ('path','sha256')})
 for p in [__file__,ENTRYPOINT,config,approval_path]:add(ref(p))
 canonical=ctx['right_context']['oldcontract']['canonical_definition'];add(canonical['source_manifest'])
 pixels._verify_sources(refs)
 return refs

def check_geometry_gate(report):
 keys=sorted(['core','skin','band',*MOUNT_KEYS]);cross={a+'__'+b for a in keys for b in keys};internal={a+'__'+b for i,a in enumerate(keys) for b in keys[i+1:]}
 if report.get('cross_pair_count')!=81 or set(report.get('cross_pairs',{}))!=cross or report.get('internal_pair_count_each')!=36 or set(report.get('internal_pairs',{}))!={'left','right'} or any(set(rows)!=internal for rows in report['internal_pairs'].values()):raise ValueError('Complete81 cross +36 each internal actor surface pairs required')
 rows=list(report['cross_pairs'].values())+[r for side in report['internal_pairs'].values() for r in side.values()]
 if any(not isinstance(r.get('outside_or_unresolved_pairs'),list) for r in rows):raise ValueError('Explicit unresolved pair lists required')
 unresolved=sum(len(r['outside_or_unresolved_pairs']) for r in rows)
 if type(report.get('outside_or_unresolved_surface_pairs')) is not int or report['outside_or_unresolved_surface_pairs']!=unresolved or unresolved:raise ValueError('Any unresolved contact or underreported list blocks pixel rendering')
 return True

def inspection_entry():
 return {'sample_id':'f0_bilateral_inspection','index':0,'seed':20261020,'scene_family':'bilateral_f0_inspection','target_count':2,'target_presence':[True,True],'group_id':'f0_bilateral_inspection_group','background_source_id':'photometry_v003_engineering_environment','background_seed':20261020,'split':'eval','frame_reference':0}

def execute_blender(config,out,threads,gpu_uuid=None,probe_only=False):
 import bpy
 stage='prepare';scene=None;contract=None;backend=None;ctx=None;c=None;refs=None;primary=None;result=None;final_errors=[];entry=inspection_entry();target_dict=None
 # Host-created immutable launch evidence admits this child; never retry it.
 out=bounded(out)
 if not out.is_relative_to(ROOT/'outputs') or out==ROOT/'outputs' or not out.is_dir():raise ValueError('Existing host-owned outputs run required')
 if (out/'started.json').exists():raise FileExistsError('Inside-Blender attempt already started; no retry')
 launch=read(out/'launch.json')
 try:
  backend,c,ctx=prepare(config);runs=run_root_from_config(c);out=output_path(out,allowed_runs=runs,fresh=False);a,approval_path,bindings=approval(config,c,out,threads=threads,gpu_uuid=gpu_uuid,probe_only=probe_only)
  if launch['approval_sha256']!=sha(approval_path) or launch['scene_config']!=ref(config) or launch['adapter_source_sha256']!=sha(__file__):raise ValueError('Host launch receipt differs before inside-Blender attempt')
  write(out/'started.json',{'status':'STARTED_ONE_BILATERAL_PREVIEW','probe_only':probe_only,'gpu_uuid':gpu_uuid,'threads':2,'bindings':bindings,'automatic_retry':False})
  refs=source_identities(config,c,ctx,bindings,approval_path)
  stage='create_scene';scene,contract=backend.create_scene(c,ctx);write(out/'scene_contract.json',contract)
  stage='geometry_diagnostics';diagnostics,_=backend.geometry_diagnostics(scene,c,ctx,contract);write(out/'geometry_diagnostics.json',diagnostics);check_geometry_gate(diagnostics)
  stage='verify_scene';scene_check=backend.verify_bilateral_scene(scene,c,ctx,contract);write(out/'scene_verification.json',scene_check);target_dict=backend.targets(scene)
  def verify(s):
   if s!=scene:raise ValueError('Callback must verify the supplied bilateral scene')
   backend.verify_bilateral_scene(s,c,ctx,contract);pixels._validate_targets(s,entry,target_dict)
  stage='pixel_probe' if probe_only else 'pixels'
  settings=dict(ctx['right_config']['render'])
  if not probe_only:settings['worker_gpu_uuid']=gpu_uuid
  pixel=pixels.render_frame(scene=scene,entry=entry,targets=target_dict,canonical=contract['canonical_definition'],K=backend.composed.actual_K(scene),data_out=out/'pixels',runtime_out=out/'runtime',output_role='diagnostic',source_identities=refs,render_settings=settings,verify_scene=verify,probe_only=probe_only)
  expected_calls=0 if probe_only else 4
  if pixel['render_calls']!=expected_calls or (not probe_only and (pixel['record'] is None or pixel['record']['planned_target_count']!=2 or [r['instance_id'] for r in pixel['record']['instances']]!=[1,2] or pixel['record']['excluded_instances'])):raise ValueError('Single bilateral diagnostic must retain both original IDs; no quota repair')
  result={'status':'PASS_BILATERAL_PIXEL_CPU_PROBE_NO_RENDER' if probe_only else 'BILATERAL_RENDER_COMPLETE_PENDING_HOST_PNG_VALIDATION','probe_only':probe_only,'rendered':not probe_only,'render_calls':pixel['render_calls'],'pixel_result':pixel,'entry':entry,'bindings':bindings,'source_identities':refs,'outside_or_unresolved_surface_pairs':0,'threads':2,'gpu_uuid':gpu_uuid,'whole_scene_inventory_restoration_claim':False,'dataset_generation_allowed':False,'wearing_accepted':False,'training_accepted':False}
 except Exception as exc:primary=(exc,traceback.format_exc())
 finally:
  if scene is not None and contract is not None:
   try:backend.verify_bilateral_scene(scene,c,ctx,contract)
   except Exception as exc:final_errors.append('Final scene verification: '+str(exc))
  try:
   if backend is not None and c is not None:backend.frozen_sources(c)
   if refs is not None:pixels._verify_sources(refs)
   else:
    check_ref(launch['scene_config'])
    if launch['adapter_source_sha256']!=sha(__file__):raise ValueError('Adapter source changed')
  except Exception as exc:final_errors.append('Final source verification: '+str(exc))
 if primary or final_errors:
  write(out/'failure.json',{'status':'FAILED_RETAINED_NO_AUTOMATIC_RETRY','stage':stage,'primary_error':str(primary[0]) if primary else None,'primary_traceback':primary[1] if primary else None,'final_verification_errors':final_errors,'probe_only':probe_only,'dataset_generation_allowed':False})
  if primary:raise primary[0]
  raise RuntimeError(final_errors)
 result['final_scene_and_source_verified']=True
 result['pending_host_png_validation']=not probe_only
 write(out/('preflight.json' if probe_only else 'render_receipt.json'),result)
 return result

def _host_runtime():
 if 'bpy' in sys.modules or Path(sys.prefix).resolve()!=(Path(__file__).resolve().parents[2]/'.venv').resolve():raise RuntimeError('Parent project .venv runtime required for complete PNG validation')
 from PIL import Image  # Existing host dependency; fail before any child launch.

def finalize_host_render(out,bindings):
 """Publish success only after the child receipt, record, sources and PNGs verify."""
 out=bounded(out)
 if not out.is_relative_to(ROOT/'outputs'):raise ValueError('Host report must stay in outputs')
 if (out/'diagnostic_report.json').exists() or (out/'failure.json').exists():raise FileExistsError('Immutable completed/failed attempt; no host retry')
 try:
  _host_runtime()
  receipt=ref(out/'render_receipt.json');result=read(out/'render_receipt.json')
  if result.get('status')!='BILATERAL_RENDER_COMPLETE_PENDING_HOST_PNG_VALIDATION' or result.get('pending_host_png_validation') is not True or result.get('probe_only') is not False or result.get('rendered') is not True or result.get('render_calls')!=4 or result.get('final_scene_and_source_verified') is not True or result.get('bindings')!=bindings:raise ValueError('Exact restored bilateral child render receipt required')
  pixel=result['pixel_result'];record_path=check_ref(pixel['record_ref']);record=read(record_path)
  if not record_path.is_relative_to(out/'runtime/samples') or record!=pixel['record'] or pixel.get('pending_host_png_validation') is not True or pixel.get('restoration_verified') is not True or pixel.get('source_identities_verified') is not True:raise ValueError('Immutable pending child pixel record/restore evidence mismatch')
  if record.get('output_role')!='diagnostic':raise ValueError('Bilateral preview only admits diagnostic records')
  names=[record['image']]
  for row in [*record['instances'],*record['excluded_instances']]:names.extend([row['mask_visible'],row['mask_amodal_expanded']])
  if any(not bounded(name).is_relative_to(out/'pixels') for name in names):raise ValueError('PNG must belong to this fresh diagnostic output')
  pixels._verify_sources(result['source_identities'])
  validation=pixels.validate_host_pngs(record)
  pixels._verify_sources(result['source_identities'])
  check_ref(receipt);check_ref(pixel['record_ref'])
  result['status']='BILATERAL_SINGLE_RGB_MASK_DIAGNOSTIC_COMPLETE_NOT_ACCEPTED';result['pending_host_png_validation']=False
  result['host_png_validation']={**validation,'child_render_receipt':receipt,'child_record':pixel['record_ref'],'immutable_child_record_retains_pending_stage_metadata':True}
  write(out/'diagnostic_report.json',result)
  return result
 except Exception as exc:
  write(out/'failure.json',{'status':'FAILED_RETAINED_NO_AUTOMATIC_RETRY','stage':'host_png_validation','primary_error':str(exc),'primary_traceback':traceback.format_exc(),'render_receipt_retained':(out/'render_receipt.json').exists(),'dataset_generation_allowed':False})
  raise

def launch(config,out,threads,gpu_uuid=None,probe_only=False):
 _host_runtime()
 backend,c,ctx=prepare(config);runs=run_root_from_config(c);out=output_path(out,allowed_runs=runs);a,approval_path,bindings=approval(config,c,out,threads=threads,gpu_uuid=gpu_uuid,probe_only=probe_only)
 binary=ctx['right_context']['neutral_config']['sources']['blender']['path'];cmd=build_command(config,out,binary,threads=threads,gpu_uuid=gpu_uuid,probe_only=probe_only)
 child=worker_environment(None if probe_only else gpu_uuid)
 for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):child[key]='2'
 child['PYTHONDONTWRITEBYTECODE']='1';child['PYTHONNOUSERSITE']='1'
 subdirs={'TMPDIR':'runtime/tmp','TEMP':'runtime/tmp','TMP':'runtime/tmp','XDG_CACHE_HOME':'runtime/cache','CUDA_CACHE_PATH':'runtime/cache/cuda','BLENDER_USER_CONFIG':'runtime/blender_user/config','BLENDER_USER_SCRIPTS':'runtime/blender_user/scripts','BLENDER_USER_DATAFILES':'runtime/blender_user/datafiles'}
 out.mkdir(parents=True)
 for sub in {'pixels','runtime',*subdirs.values()}:
  p=bounded(out/sub)
  if not p.is_relative_to(out):raise ValueError('Output runtime child escapes assigned run')
  p.mkdir(parents=True,exist_ok=True)
 for key,sub in subdirs.items():child[key]=str(out/sub)
 write(out/'launch.json',{'command':cmd,'scene_config':ref(config),'approval_sha256':sha(approval_path),'adapter_source_sha256':sha(__file__),'threads':2,'gpu_uuid':gpu_uuid,'probe_only':probe_only,'bindings':bindings,'automatic_retry':False})
 try:
  with (out/'blender.log').open('xb') as log:p=subprocess.run(cmd,cwd=ROOT,env=child,stdout=log,stderr=subprocess.STDOUT)
  if p.returncode:raise RuntimeError('Preserved failed bilateral diagnostic '+str(out/'blender.log'))
  return read(out/'preflight.json') if probe_only else finalize_host_render(out,bindings)
 except Exception as exc:
  write(out/'launch_failure.json',{'status':'FAILED_RETAINED_NO_RETRY','error':str(exc),'traceback':traceback.format_exc(),'automatic_retry':False});raise

def main():
 args=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output');p.add_argument('--threads',type=int);p.add_argument('--gpu-uuid');p.add_argument('--probe-only',action='store_true');p.add_argument('--prepare-only',action='store_true');p.add_argument('--inside-blender',action='store_true',help=argparse.SUPPRESS);a=p.parse_args(args)
 if a.prepare_only:
  if a.output or a.gpu_uuid or a.probe_only or a.inside_blender:raise ValueError('Prepare only cannot mix execution flags')
  prepare(a.config);print('PREPARED_ONLY_NO_SCENE_GPU_OR_RENDER');return
 if not a.output or a.threads!=2:raise ValueError('Explicit output and CPU2 required')
 if a.inside_blender and 'bpy' not in sys.modules:raise ValueError('Internal Blender mode only')
 result=(execute_blender if a.inside_blender else launch)(a.config,a.output,a.threads,a.gpu_uuid,a.probe_only)
 print(json.dumps({'status':result['status'],'rendered':result['rendered'],'render_calls':result['render_calls'],'dataset_generation_allowed':False}))
if __name__=='__main__':main()
