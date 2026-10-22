"""Fixed-scene candidate renderer; geometry checked once before costly RGB.
No training acceptance or automatic production release. Outputs are immutable.
"""
from pathlib import Path
import hashlib,json,time,os,traceback
ROOT=Path('/data/zy/vln/a-test')

def bounded(value):
    p=Path(value);p=(p if p.is_absolute() else ROOT/p).resolve()
    if not p.is_relative_to(ROOT):raise ValueError('Project boundary')
    return p

def digest_file(path):return hashlib.sha256(bounded(path).read_bytes()).hexdigest()
def reference(path):return {'path':str(bounded(path).relative_to(ROOT)),'sha256':digest_file(path)}
def read(path):return json.loads(bounded(path).read_text())
def write(path,value):
    with bounded(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)

def validate_config(config,plan):
    from data.action4_devices import ALLOWED_GPU_UUIDS
    if config.get('schema')!='action4.template_worker.v1' or plan.get('schema')!='action4.fixed_template_batch_plan.v1':raise ValueError('Explicit template worker and plan required')
    ids=config['indices']
    if not ids or len(ids)!=len(set(ids)) or any(type(i) is not int or i<0 or i>=plan['preview_count'] for i in ids):raise ValueError('Unique preview-only indices required')
    if any(plan['entries'][i]['template']!=config['template'] for i in ids):raise ValueError('One fixed scene per worker')
    if config['gpu_uuid'] not in ALLOWED_GPU_UUIDS or config['threads']!=2:raise ValueError('Authorized GPU and two-thread worker required')
    if type(config['max_attempts']) is not int or not 1<=config['max_attempts']<=64:raise ValueError('Bounded retry count required')
    data=bounded(config['data_out']);runtime=bounded(config['runtime_out'])
    if not data.is_relative_to(ROOT/'data/action4_synth') or not runtime.is_relative_to(ROOT/'outputs') or runtime==ROOT/'outputs':raise ValueError('Candidate data and runtime output roles required')
    for i in ids:
        e=plan['entries'][i]['entry'];presence=e['target_presence']
        if len(presence)!=2 or any(type(v) is not bool for v in presence) or sum(presence)!=e['target_count'] or e['split']=='test' and presence!=[True,True]:raise ValueError('Exact presence and dual-only test required')
    return data,runtime

def run_blender(config_path):
    if Path.cwd().resolve()!=ROOT:raise ValueError('Exact working directory required')
    config=read(config_path);plan=read(config['plan_path']);data,out=validate_config(config,plan)
    if data.exists() or out.exists():raise FileExistsError('Existing output belongs to a previous invocation; untouched')
    out.mkdir(parents=True,exist_ok=False)
    try:
        data.mkdir(parents=True,exist_ok=False)
        return _run_blender_owned(config_path,config,plan,data,out)
    except BaseException:
        write(out/'FAILED.json',{'status':'FAILED_OWNED_INVOCATION','traceback':traceback.format_exc(),'training_accepted':False})
        raise

def _run_blender_owned(config_path,config,plan,data,out):
    import bpy
    from data import action4_parameter_sampling as sampling
    from data import action4_modular_pixels as pixels
    from data.action4_template_runtime import TemplateRuntime
    from data.action4_template_collision import TemplateCollisionChecker
    if Path.cwd().resolve()!=ROOT:raise ValueError('Exact working directory required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=config['gpu_uuid']:raise ValueError('Worker visibility differs from assigned GPU')
    started=time.monotonic();runtime=TemplateRuntime(config['template']);checker=TemplateCollisionChecker(runtime)
    policy=read(plan['policy_path']);canonical=runtime.metadata['canonical_definition'];source_paths=[
        'src/data/action4_template_batch.py','scripts/action4_render_template.py','src/data/action4_template_runtime.py',
        'src/data/action4_template_collision.py','src/data/action4_parameter_sampling.py','src/data/action4_modular_assembly.py',
        'src/data/action4_modular_bilateral_scene.py','src/data/action4_mount_modules.py','src/data/action4_devices.py',config_path,config['plan_path'],
        plan['policy_path'],plan['layout_path'],runtime.metadata_path,runtime.metadata['blend_ref']['path'],
        runtime.metadata['baseline_parameters_ref']['path'],runtime.metadata['parent_manifest_ref']['path'],canonical['source_manifest']['path'],
        *pixels.REQUIRED_SOURCE_PATHS,*[x['path'] for x in runtime.additional_source_refs]]
    source_ids={str(bounded(path).relative_to(ROOT)):digest_file(path) for path in source_paths}
    write(out/'execution_sources.json',{'schema':'action4.template_batch_sources.v1','source_identities':source_ids,'template':runtime.name,'template_loads':1,'fixed_environment_layout':plan['layout_path'],'production_accepted':False})
    execution_ref=reference(out/'execution_sources.json')
    pixel_ids={path:source_ids[path] for path in pixels.REQUIRED_SOURCE_PATHS};pixel_ids[canonical['source_manifest']['path']]=canonical['source_manifest']['sha256']
    if source_ids[canonical['source_manifest']['path']]!=canonical['source_manifest']['sha256']:raise ValueError('Canonical manifest drift')
    accepted=[];rejected=0;rendered=0
    with (out/'attempts.jsonl').open('x') as journal,(data/'manifest.jsonl').open('x') as manifest:
        for index in config['indices']:
            planned=plan['entries'][index]['entry'];found=False
            for attempt in range(config['max_attempts']):
                draw_index=index+attempt*plan['count'];entry=dict(planned);entry.update(sample_id=planned['sample_id']+f'_a{attempt:02}',seed=plan['seed']+draw_index)
                vector=sampling.sample_uniform(draw_index,plan['seed'])
                proposal=sampling.propose(runtime.anchor,entry,policy,vector,canonical['corners_3d_m'],anchor_ref=runtime.metadata['baseline_parameters_ref'],template_name=runtime.name)
                targets=runtime.apply(proposal);geometry=checker.check()
                geometry.update(sample_id=entry['sample_id'],proposal_digest=sampling.digest(proposal),plan_index=index,generation_attempt=attempt)
                info={'plan_index':index,'planned_sample_id':planned['sample_id'],'attempt':attempt,'sample_id':entry['sample_id'],'draw_index':draw_index,'collision':geometry}
                if not geometry['accepted']:
                    info['status']='REJECT_GEOMETRY';journal.write(json.dumps(info)+'\n');journal.flush();rejected+=1;continue
                actual=runtime.export_corners()
                settings={'device':'CUDA','denoiser':'OPTIX','samples':64,'mask_samples':1,'amodal_mode':'crop','amodal_margin_px':16,'seed':entry['seed'],'worker_gpu_uuid':config['gpu_uuid']}
                begin=time.monotonic()
                result=pixels.render_frame(scene=runtime.scene,entry=entry,targets=targets,canonical=canonical,K=runtime.K.tolist(),data_out=data,runtime_out=out,output_role='dataset_candidate',source_identities=pixel_ids,render_settings=settings,verify_scene=runtime.verify)
                rendered+=1;record=result['record'];info['render_seconds']=time.monotonic()-begin
                expected=[j for j,on in enumerate(entry['target_presence'],1) if on]
                observed=[row['instance_id'] for row in record['instances']]
                if record['excluded_instances'] or observed!=expected or any(row.get('ignore',False) for row in record['instances']):
                    info.update(status='REJECT_NO_TRAINING_SUPPORT',record_ref=result['record_ref']);journal.write(json.dumps(info)+'\n');journal.flush();rejected+=1;continue
                sample=out/'samples'/entry['sample_id'];write(sample/'parameters.json',proposal);write(sample/'collision.json',geometry)
                enriched={**record,'template_execution_ref':execution_ref,'template_parameters_ref':reference(sample/'parameters.json'),'template_collision_ref':reference(sample/'collision.json'),'planned_sample_id':planned['sample_id'],'plan_index':index,'generation_attempt':attempt,'template_actual_projection_error_px':actual['max_proposal_vs_actual_error_px']}
                write(sample/'candidate_record.json',enriched);manifest.write(json.dumps(enriched)+'\n');manifest.flush()
                info.update(status='CANDIDATE_RENDERED_PENDING_FINAL_HOST',record_ref=reference(sample/'candidate_record.json'));journal.write(json.dumps(info)+'\n');journal.flush();accepted.append(info);found=True;break
            if not found:raise RuntimeError('No admissible candidate within attempt budget for index '+str(index))
    for path,expected in source_ids.items():
        if digest_file(path)!=expected:raise ValueError('Source changed during batch: '+path)
    report={'status':'CANDIDATE_BATCH_COMPLETE_FINAL_HOST_PENDING','template':runtime.name,'template_loads':1,'count':len(accepted),'rejected_attempts':rejected,'rgb_frames_rendered':rendered,'entries':accepted,'manifest_ref':reference(data/'manifest.jsonl'),'execution_sources_ref':execution_ref,'worker_config_ref':reference(config_path),'plan_ref':reference(config['plan_path']),'elapsed_seconds':time.monotonic()-started,'training_accepted':False,'final_host_complete':False}
    write(out/'COMPLETE.json',report);return report
