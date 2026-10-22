"""Explicit persistent-scene launcher using the frozen Action4 plan."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time
from . import action4_synthesis as synth
from . import action4_execution_authority as authority
from .action4_execution_cache import FastHostSession
from .action4_devices import worker_environment
ROOT=synth.ROOT

def policy_inputs(plan_path,output,shard_index,num_shards,entry_start,entry_count,limit):
    refs=authority.authorize_render(ROOT,'scripts/action4_render_fast.py',plan_path,output,shard_index,num_shards,entry_start,entry_count,limit)
    current,_=authority.active(ROOT);policy_ref=current['render']['execution_policy_ref']
    policy=authority.readref(ROOT,policy_ref)
    return current['render']['plan_ref'],policy_ref,policy,refs

def command(blender,plan,output,threads,shard_index,num_shards,gpu_uuid,entry_start,entry_count,limit):
    argv=[str(blender),'--factory-startup','--disable-autoexec','-noaudio','--background','--threads',str(threads),'--python-exit-code','1','--python',str(ROOT/'scripts/action4_render_fast.py'),'--','_render-shard','--plan',str(plan),'--output',str(output),'--threads',str(threads),'--shard-index',str(shard_index),'--num-shards',str(num_shards),'--entry-start',str(entry_start)]
    for name,value in [('gpu-uuid',gpu_uuid),('entry-count',entry_count),('limit',limit)]:
        if value is not None:argv.extend(['--'+name,str(value)])
    return argv

def render(a):
    if Path.cwd().resolve()!=ROOT:raise ValueError('Exact project root required')
    pr,policy_ref,policy,refs=policy_inputs(a.plan,a.output,a.shard_index,a.num_shards,a.entry_start,a.entry_count,a.limit)
    plan=authority.readref(ROOT,pr);out=synth._path(a.output);run=synth._run_output_dir(out)
    if out.exists() or run.exists() or not out.is_relative_to(ROOT/'data'):raise ValueError('Fresh data and runtime shard required')
    chosen=synth.select_shard_entries(plan,a.shard_index,a.num_shards,a.limit,a.entry_start,a.entry_count)
    # Existing completed sample IDs are immutable and must not be regenerated.
    for manifest in out.parent.glob('*/manifest.jsonl'):
        path=synth._path(manifest,exists=True)
        existing={json.loads(line)['sample_id'] for line in path.read_text().splitlines() if line.strip()}
        if existing.intersection(e['sample_id'] for e in chosen):raise ValueError('Selected samples already rendered')
    if plan['recipe']['render']['device']!='CPU' and a.gpu_uuid not in plan['recipe']['render'].get('gpu_uuids',[]):raise ValueError('Unapproved GPU')
    env=worker_environment(a.gpu_uuid);out.mkdir(parents=True);run.mkdir(parents=True)
    for key,name in [('TMPDIR','tmp'),('XDG_CACHE_HOME','cache'),('BLENDER_USER_CONFIG','blender_config'),('BLENDER_USER_SCRIPTS','blender_scripts'),('BLENDER_USER_DATAFILES','blender_data')]:
        p=run/name;p.mkdir();env[key]=str(p)
    env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS=str(a.threads),OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    cmd=command(synth._path(a.blender,exists=True),synth._path(a.plan,exists=True),out,a.threads,a.shard_index,a.num_shards,a.gpu_uuid,a.entry_start,a.entry_count,a.limit)
    synth._write(run/'launch.json',{'command':cmd,'gpu_uuid':a.gpu_uuid,'entry_start':a.entry_start,'entry_count':a.entry_count,'expected_sample_ids':[e['sample_id'] for e in chosen],'plan_sha256':pr['sha256'],'execution_policy_ref':policy_ref,'execution_authority_refs':refs,'metadata_validation':'exact_create_plan_reuse_fresh_boundaries_original_record_checks','persistent_base':True})
    start=time.perf_counter()
    with (run/'blender.log').open('xb') as log:
        completed=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    render_seconds=time.perf_counter()-start
    if completed.returncode:raise RuntimeError('Blender failed; inspect '+str(run/'blender.log'))
    start=time.perf_counter()
    with FastHostSession(pr,handoff_ref=policy['handoff_ref']) as host:
        report=synth.validate_shard(out,validation_session=host)
    if report['sample_ids']!=[e['sample_id'] for e in chosen]:raise ValueError('Completed IDs differ')
    for ref in refs:
        if authority.ref(ROOT,ref['path'])!=ref:raise ValueError('Execution dependency changed')
    synth._write(run/'fast_launch_timing.json',{'render_worker_seconds':render_seconds,'independent_host_seconds':time.perf_counter()-start,'images':len(chosen),'training_accepted':False})
    return report

def main(argv=None):
    if argv is None:argv=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('command',choices=['render-shard','_render-shard'])
    p.add_argument('--plan',required=True);p.add_argument('--output',required=True);p.add_argument('--blender')
    p.add_argument('--threads',type=int,required=True);p.add_argument('--shard-index',type=int,default=0);p.add_argument('--num-shards',type=int,default=1)
    p.add_argument('--gpu-uuid');p.add_argument('--entry-start',type=int,default=0);p.add_argument('--entry-count',type=int);p.add_argument('--limit',type=int)
    a=p.parse_args(argv)
    if not 1<=a.threads<=32:raise ValueError('Explicit threads 1..32 required')
    if a.command=='render-shard':
        if not a.blender:raise ValueError('Project-local Blender required')
        result=render(a)
    else:
        import bpy
        from .action4_fast_generation import render_inside_blender
        pr,policy_ref,policy,refs=policy_inputs(a.plan,a.output,a.shard_index,a.num_shards,a.entry_start,a.entry_count,a.limit)
        result=render_inside_blender(a.plan,a.output,a.shard_index,a.num_shards,a.limit,gpu_uuid=a.gpu_uuid,entry_start=a.entry_start,entry_count=a.entry_count,handoff_ref=policy['handoff_ref'],execution_policy_ref=policy_ref)
    print(json.dumps(result,ensure_ascii=False,indent=2));return 0
