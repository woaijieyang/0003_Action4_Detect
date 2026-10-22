"""Cached metadata launch for the unchanged Action4 renderer and Host validator.
One owner-bound index per launch, never a persisted trusted flag. No geometry,
label, source-context, renderer or Host acceptance check is removed.
"""
from __future__ import annotations
import argparse,hashlib,json,subprocess,time
from pathlib import Path
from . import action4_synthesis as synthesis
from . import action4_modular_synthesis_backend as backend
ROOT=synthesis.ROOT

def validate_launch_plan(plan_path):
 start=time.perf_counter();path=synthesis._path(plan_path,exists=True);plan_ref={'path':str(path.relative_to(ROOT)),'sha256':synthesis._hash(path)}
 index=backend._PlanIndex();plan=index.read_ref(plan_ref);index.bind_plan(plan)
 if index.read_ref({'path':plan['recipe_path'],'sha256':plan['recipe_sha256']})!=plan['recipe']:raise ValueError('Serialized recipe differs from plan')
 if plan.get('status')!='render_candidates_pending_acceptance' or plan['recipe'].get('asset_backend')!='modular_composed':raise ValueError('Frozen modular candidate plan required')
 synthesis.validate_recipe(plan['recipe'],_validation_index=index)
 if plan.get('synthesis_code_sha256')!=index.hash_path(synthesis.__file__) or plan.get('render_base_sha256')!=index.hash_path(ROOT/'src/data/action4_render.py'):raise ValueError('Original renderer changed after plan freeze')
 if plan.get('asset_identities')!=synthesis._asset_identities(plan['recipe'],_validation_index=index):raise ValueError('Frozen asset identities changed')
 backend.validate_plan(plan,_index=index)
 counts=dict(index.events);index.seal()
 return plan,plan_ref,{'status':'PASS_OWNER_SCOPED_LAUNCH_METADATA_ONLY','plan_ref':plan_ref,'opening_counts':counts,'elapsed_seconds':time.perf_counter()-start,'original_Blender_and_Host_validation_still_required':True,'training_accepted':False}

def validate_launch_identity(plan_path,expected_ref):
 """Approved immutable plan identity only; original child full checks remain.
 This fast route is pinned to the v006 renderer/backend whose RenderBatch
 constructs ValidatedPlanContext before any per-entry rendering. It does not
 manufacture a validated context, skip child checks or trust a PASS flag.
 """
 start=time.perf_counter();path=synthesis._path(plan_path,exists=True)
 if str(path.relative_to(ROOT))!=expected_ref['path']:raise ValueError('Authority plan path mismatch')
 index=backend._PlanIndex();plan=index.read_ref(expected_ref);index.bind_plan(plan)
 if plan.get('status')!='render_candidates_pending_acceptance' or plan['recipe'].get('asset_backend')!='modular_composed':raise ValueError('Frozen modular candidate plan required')
 if plan.get('synthesis_code_sha256')!='b653187fb1153437ed9b61999c7025d7d7ff47c76f16001d2a2b2eb81c76a8df' or plan['asset_identities'].get('src/data/action4_modular_synthesis_backend.py')!='b8906399efa54fe47117ed458a4487ab253ef810aaf536f174f1bd6c98529d41':raise ValueError('Original child validation contract changed; review new version explicitly')
 if index.read_ref({'path':plan['recipe_path'],'sha256':plan['recipe_sha256']})!=plan['recipe']:raise ValueError('Serialized recipe differs from frozen plan')
 if index.hash_path(synthesis.__file__)!=plan['synthesis_code_sha256'] or index.hash_path(ROOT/'src/data/action4_render.py')!=plan['render_base_sha256']:raise ValueError('Original renderer source drift')
 for name,digest in plan['asset_identities'].items():index.check_ref({'path':name,'sha256':digest})
 counts=dict(index.events);index.seal()
 return plan,dict(expected_ref),{'status':'PASS_APPROVED_PLAN_IDENTITIES_CHILD_FULL_VALIDATION_REQUIRED','plan_ref':dict(expected_ref),'opening_counts':counts,'elapsed_seconds':time.perf_counter()-start,'original_Blender_and_Host_validation_still_required':True,'semantic_checks_reused_from':'root-approved exact create_plan output; original RenderBatch and Host independently retain full checks','training_accepted':False}

def make_command(plan_path,output_dir,blender,shard_index,num_shards,threads,gpu_uuid,entry_start,entry_count,limit):
 command=[str(blender),'--factory-startup','--disable-autoexec','-noaudio','--background','--threads',str(threads),'--python-exit-code','1','--python',str(ROOT/'scripts/action4_synthesis.py'),'--','_render-shard','--plan',str(plan_path),'--output',str(output_dir),'--shard-index',str(shard_index),'--num-shards',str(num_shards)]
 if limit is not None:command.extend(['--limit',str(limit)])
 command.extend(['--entry-start',str(entry_start)])
 if entry_count is not None:command.extend(['--entry-count',str(entry_count)])
 if gpu_uuid is not None:command.extend(['--gpu-uuid',gpu_uuid])
 return command

def render_shard(plan_path,output_dir,blender,shard_index=0,num_shards=1,limit=None,threads=2,*,gpu_uuid=None,entry_start=0,entry_count=None):
 if Path.cwd().resolve()!=ROOT:raise ValueError('Exact project root required')
 from .action4_execution_authority import authorize_render
 authority_refs=authorize_render(ROOT,'scripts/action4_render_cached.py',plan_path,output_dir,shard_index,num_shards,entry_start,entry_count,limit)
 plan_path=synthesis._path(plan_path,exists=True);output_dir=synthesis._path(output_dir);blender=synthesis._path(blender,exists=True)
 if not output_dir.is_relative_to(ROOT/'data') or output_dir==ROOT/'data' or output_dir.exists():raise ValueError('Fresh project data shard required')
 if type(threads) is not int or not 1<=threads<=64:raise ValueError('Explicit valid Blender thread budget')
 run_dir=synthesis._run_output_dir(output_dir)
 if run_dir.exists():raise FileExistsError('Fresh runtime directory required')
 expected_ref=next(r for r in authority_refs if r['path']==str(plan_path.relative_to(ROOT)))
 plan,plan_ref,preflight=validate_launch_identity(plan_path,expected_ref)
 chosen=synthesis.select_shard_entries(plan,shard_index,num_shards,limit,entry_start,entry_count)
 from .action4_devices import worker_environment
 settings=plan['recipe']['render']
 if settings['device']=='CPU' and gpu_uuid is not None:raise ValueError('CPU plan cannot switch to GPU')
 if settings['device']!='CPU' and gpu_uuid not in settings.get('gpu_uuids',[]):raise ValueError('GPU UUID outside frozen plan allowlist')
 # Fresh worker environment and original inner CLI; no direct queue access.
 env=worker_environment(gpu_uuid)
 if synthesis._hash(plan_path)!=plan_ref['sha256']:raise ValueError('Plan changed before Blender launch')
 output_dir.mkdir(parents=True);run_dir.mkdir(parents=True,exist_ok=False)
 command=make_command(plan_path,output_dir,blender,shard_index,num_shards,threads,gpu_uuid,entry_start,entry_count,limit)
 launcher_ref={'path':str(Path(__file__).resolve().relative_to(ROOT)),'sha256':synthesis._hash(__file__)}
 synthesis._write(run_dir/'launch_preflight.json',preflight)
 synthesis._write(run_dir/'launch.json',{'command':command,'gpu_uuid':gpu_uuid,'entry_start':entry_start,'entry_count':entry_count,'expected_sample_ids':[e['sample_id'] for e in chosen],'plan_sha256':plan_ref['sha256'],'launcher_ref':launcher_ref,'execution_authority_refs':authority_refs,'metadata_validation':'approved_identity_preflight_original_child_full_validation','worker_and_Host_checks':'unchanged original entrypoints'})
 with (run_dir/'blender.log').open('xb') as log:
  process=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=False)
 if process.returncode:raise RuntimeError(f'Blender exited {process.returncode}; inspect {run_dir / "blender.log"}')
 # Host still owns independent geometry/PNG/source closing and atomic reports.
 report=synthesis.validate_shard(output_dir)
 if report['sample_ids']!=[e['sample_id'] for e in chosen]:raise ValueError('Completed IDs differ from frozen shard request')
 from .action4_execution_authority import readref
 for reference in authority_refs:readref(ROOT,reference)
 return report

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);p.add_argument('command',choices=['render-shard']);p.add_argument('--plan',required=True);p.add_argument('--output',required=True);p.add_argument('--blender',required=True);p.add_argument('--shard-index',type=int,default=0);p.add_argument('--num-shards',type=int,default=1);p.add_argument('--limit',type=int);p.add_argument('--threads',type=int,required=True);p.add_argument('--gpu-uuid');p.add_argument('--entry-start',type=int,default=0);p.add_argument('--entry-count',type=int);a=p.parse_args(argv)
 print(json.dumps(render_shard(a.plan,a.output,a.blender,a.shard_index,a.num_shards,a.limit,a.threads,gpu_uuid=a.gpu_uuid,entry_start=a.entry_start,entry_count=a.entry_count),indent=2));return 0
